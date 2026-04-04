import sys
import os
import gc
import torch
import cv2
import numpy as np
import time
import unicodedata
from loguru import logger

from src.modules.ocr.detector import TextDetector
from src.modules.ocr.recognizer import TextRecognizer
from src.modules.ocr.cropper import crop_and_rotate_box
from src.modules.ocr.post_process import clean_vba_text

torch.set_num_threads(4)


class OCREngine:
    def __init__(self, use_gpu: bool = False):
        self.detector = TextDetector(use_gpu=use_gpu)
        self.recognizer = TextRecognizer(use_gpu=use_gpu)

    def detect_and_normalize(self, img_array: np.ndarray) -> dict:
        """
        Detect text boxes and normalize image orientation so returned image is upright
        and boxes are horizontal when possible. Performs a quick 180-degree "taste test"
        (sample top 3-5 boxes and check recognizer confidence) and flips 180° if the
        recognizer struggles (common when page is upside-down).

        Returns: {"working_img": np.ndarray, "boxes": list, "detection_time_ms": int, "rotated_angle": int}
        """
        try:
            original_img = img_array.copy()
            if len(original_img.shape) == 2:
                original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)

            working_img = original_img
            rotated_angle = 0

            def _local_detect(img):
                try:
                    t0 = time.time()
                    boxes = self.detector.detect(img)
                    t1 = time.time()
                    det_ms = int((t1 - t0) * 1000)
                    boxes = boxes or []
                    return boxes, det_ms, len(boxes)
                except Exception as e:
                    logger.exception(f"detect_and_normalize: detection failed: {e}")
                    return [], 0, 0

            def _count_orientation(boxes_list):
                v = 0
                h = 0
                for b in boxes_list:
                    try:
                        pts = np.array(b, dtype=np.float32)
                        w = int(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3])))
                        hh = int(max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2])))
                        if hh > w * 1.2:
                            v += 1
                        elif w > hh * 1.2:
                            h += 1
                    except Exception:
                        continue
                return v, h

            # Initial detection
            boxes, det_ms, _ = _local_detect(working_img)
            vertical_boxes, horizontal_boxes = _count_orientation(boxes)
            logger.debug(f"detect_and_normalize geometry: vertical={vertical_boxes}, horizontal={horizontal_boxes}")

            # If vertical boxes dominate, try rotating to produce horizontal boxes
            if vertical_boxes > horizontal_boxes:
                # Try 90 degrees CCW first
                trial_img = cv2.rotate(working_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                t_boxes, t_det_ms, _ = _local_detect(trial_img)
                t_v, t_h = _count_orientation(t_boxes)
                if t_h > t_v and t_boxes:
                    working_img = trial_img
                    boxes = t_boxes or []
                    det_ms = t_det_ms
                    rotated_angle = (rotated_angle + 90) % 360
                else:
                    # Try 270 (90 CW)
                    trial_img2 = cv2.rotate(working_img, cv2.ROTATE_90_CLOCKWISE)
                    t2_boxes, t2_det_ms, _ = _local_detect(trial_img2)
                    t2_v, t2_h = _count_orientation(t2_boxes)
                    if t2_h > t2_v and t2_boxes:
                        working_img = trial_img2
                        boxes = t2_boxes or []
                        det_ms = t2_det_ms
                        rotated_angle = (rotated_angle + 270) % 360
                    else:
                        boxes = boxes or []
            else:
                boxes = boxes or []

            # Prefer horizontal boxes if available
            try:
                horiz_boxes = []
                for b in (boxes or []):
                    try:
                        pts = np.array(b, dtype=np.float32)
                        w = int(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3])))
                        hh = int(max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2])))
                        if w > hh * 1.2:
                            horiz_boxes.append(b)
                    except Exception:
                        continue
                if horiz_boxes:
                    boxes_out = horiz_boxes
                else:
                    boxes_out = boxes or []
            except Exception:
                boxes_out = boxes or []

            # QUICK TASTE TEST for upside-down pages (180-degree check)
            try:
                sample_boxes = []
                if boxes_out and len(boxes_out) >= 3:
                    # sort boxes by their vertical center (top-most first)
                    def _cy(box):
                        try:
                            pts = np.array(box, dtype=np.float32)
                            return float((pts[0][1] + pts[1][1] + pts[2][1] + pts[3][1]) / 4.0)
                        except Exception:
                            return 1e9

                    sorted_boxes = sorted(boxes_out, key=_cy)
                    top_n = min(5, len(sorted_boxes))
                    sample_boxes = sorted_boxes[:top_n]

                if sample_boxes:
                    quick_res = self.recognize_from_boxes(working_img, sample_boxes, detection_time_ms=det_ms)
                    # Combine recognized lines into a single sample string
                    sample_text = quick_res.get('text', '') or ' '.join(quick_res.get('text_lines', []))
                    sample_text_lower = (sample_text or '').lower()

                    def _norm_ascii(s: str) -> str:
                        try:
                            s2 = unicodedata.normalize('NFKD', s)
                            s2 = ''.join(ch for ch in s2 if not unicodedata.combining(ch))
                            return s2.lower()
                        except Exception:
                            return s.lower() if s else ''

                    norm_sample = _norm_ascii(sample_text_lower)

                    # Vietnamese administrative keywords (longer multi-syllable phrases only)
                    keywords = ['cộng hòa', 'độc lập', 'quyết định', 'công ty', 'ủy ban', 'nhân dân', 'giám đốc', 'điều lệ', 'thành phố', 'phê duyệt']
                    norm_keywords = [_norm_ascii(k) for k in keywords]

                    found = False
                    for k in keywords:
                        if k in sample_text_lower:
                            found = True
                            break
                    if not found:
                        for nk in norm_keywords:
                            if nk in norm_sample:
                                found = True
                                break

                    # If none of the keywords were found, the sample is likely upside-down/gibberish
                    if not found:
                        logger.info("🔍 Dictionary check failed on header sample -> rotating 180° to correct upside-down text")
                        rot180 = cv2.rotate(working_img, cv2.ROTATE_180)
                        re_boxes, re_det_ms, _ = _local_detect(rot180)
                        # prefer horizontal boxes from re-detection
                        try:
                            re_horiz = []
                            for b in (re_boxes or []):
                                try:
                                    pts = np.array(b, dtype=np.float32)
                                    w = int(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3])))
                                    hh = int(max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2])))
                                    if w > hh * 1.2:
                                        re_horiz.append(b)
                                except Exception:
                                    continue
                            if re_horiz:
                                boxes_out = re_horiz
                            else:
                                boxes_out = re_boxes or []
                        except Exception:
                            boxes_out = re_boxes or []

                        working_img = rot180
                        det_ms = re_det_ms or det_ms
                        rotated_angle = (rotated_angle + 180) % 360

            except Exception as e:
                logger.exception(f"Quick taste test failed: {e}")

            return {"working_img": working_img, "boxes": boxes_out, "detection_time_ms": int(det_ms or 0), "rotated_angle": int(rotated_angle)}

        except Exception as e:
            logger.exception(f"detect_and_normalize error: {e}")
            return {"working_img": img_array, "boxes": [], "detection_time_ms": 0, "rotated_angle": 0}

    def recognize_from_boxes(self, img: np.ndarray, boxes: list, detection_time_ms: int = 0) -> dict:
        """
        Recognize text only for the provided boxes. Always returns a safe dictionary.
        """
        try:
            boxes_count = len(boxes) if boxes is not None else 0
            if not boxes:
                return {"text_lines": [], "text": "", "confidence": 0.0, "detection_time_ms": int(detection_time_ms or 0), "recognition_time_ms": 0, "boxes_count": boxes_count, "valid_boxes_count": 0}

            valid_boxes = []
            cropped_images = []
            for box in boxes:
                try:
                    pts = np.array(box, dtype=np.float32)
                    w = int(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3])))
                    h = int(max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2])))
                    if w < 10 or h < 10 or h > w * 5:
                        continue
                    cropped_images.append(crop_and_rotate_box(img, box))
                    valid_boxes.append(box)
                except Exception:
                    continue

            valid_boxes_count = len(valid_boxes)
            logger.debug(f"🔍 Recognize_from_boxes: boxes={boxes_count}, valid={valid_boxes_count}")
            if not valid_boxes:
                return {"text_lines": [], "text": "", "confidence": 0.0, "detection_time_ms": int(detection_time_ms or 0), "recognition_time_ms": 0, "boxes_count": boxes_count, "valid_boxes_count": valid_boxes_count}

            rec_start = time.time()
            texts, probs = self.recognizer.recognize_batch(cropped_images)
            rec_end = time.time()
            recognition_time_ms = int((rec_end - rec_start) * 1000)

            extracted_items = []
            for i, box in enumerate(valid_boxes):
                try:
                    clean_text = clean_vba_text(texts[i])
                    if clean_text:
                        cy = (box[0][1] + box[3][1]) / 2
                        left_x = min(box[0][0], box[3][0])
                        prob = probs[i] if probs and i < len(probs) else 0.0
                        extracted_items.append({"text": clean_text, "cy": cy, "left_x": left_x, "prob": prob})
                except Exception:
                    continue

            # cleanup
            try:
                del cropped_images
            except Exception:
                pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if not extracted_items:
                return {"text_lines": [], "text": "", "confidence": 0.0, "detection_time_ms": int(detection_time_ms or 0), "recognition_time_ms": recognition_time_ms, "boxes_count": boxes_count, "valid_boxes_count": valid_boxes_count}

            extracted_items.sort(key=lambda item: item['cy'])
            lines = []
            current_line = [extracted_items[0]]
            for item in extracted_items[1:]:
                if abs(item['cy'] - current_line[-1]['cy']) < 15:
                    current_line.append(item)
                else:
                    lines.append(current_line)
                    current_line = [item]
            lines.append(current_line)

            final_text_lines = []
            total_prob = 0
            for line in lines:
                line.sort(key=lambda item: item['left_x'])
                line_text = " | ".join([item['text'] for item in line])
                final_text_lines.append(line_text)
                total_prob += sum([item['prob'] for item in line])

            confidence = round((total_prob / len(extracted_items)) * 100, 2)
            return {
                "text_lines": final_text_lines,
                "text": "\n".join(final_text_lines),
                "confidence": confidence,
                "detection_time_ms": int(detection_time_ms or 0),
                "recognition_time_ms": recognition_time_ms,
                "boxes_count": boxes_count,
                "valid_boxes_count": valid_boxes_count
            }
        except Exception as e:
            logger.exception(f"recognize_from_boxes error: {e}")
            return {"text_lines": [], "text": "", "confidence": 0.0, "detection_time_ms": int(detection_time_ms or 0), "recognition_time_ms": 0, "boxes_count": 0, "valid_boxes_count": 0}

    def extract(self, img_array: np.ndarray, fast_mode: bool = False) -> dict:
        """
        Higher-level extraction that uses detect_and_normalize + recognize_from_boxes and
        falls back to simple rotation retries when needed. Returns a safe dict similar
        to recognize_from_boxes but may attempt a few guided rotation retries.
        """
        try:
            original_img = img_array.copy()
            if len(original_img.shape) == 2:
                original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)

            # Orientation hint from detector (not authoritative)
            try:
                angle, score = self.detector.get_orientation(original_img)
            except Exception:
                angle, score = 0, 0.0

            rotated_angle = 0
            ROTATE_CONF_THRESHOLD = 0.98
            MIN_ROTATE_TRY = 0.6

            # If detector is very confident, pre-rotate
            if score > ROTATE_CONF_THRESHOLD and angle in (90, 180, 270):
                logger.info(f"🔄 AI detected orientation {angle}° (score {score:.2f}), applying pre-rotate")
                if angle == 90:
                    working_img = cv2.rotate(original_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif angle == 180:
                    working_img = cv2.rotate(original_img, cv2.ROTATE_180)
                else:
                    working_img = cv2.rotate(original_img, cv2.ROTATE_90_CLOCKWISE)
                rotated_angle = angle
            else:
                working_img = original_img

            # Use detect_and_normalize to get upright working image and boxes
            dn = self.detect_and_normalize(working_img)
            working_img = dn.get('working_img', working_img)
            boxes = dn.get('boxes', []) or []
            detection_time_ms = int(dn.get('detection_time_ms', 0) or 0)
            rotated_angle = (rotated_angle + int(dn.get('rotated_angle', 0) or 0)) % 360

            # Recognize from boxes
            base_result = self.recognize_from_boxes(working_img, boxes, detection_time_ms)

            if fast_mode:
                base_result['rotated_angle'] = int(rotated_angle or 0)
                return base_result

            # If confidence is low, try guided rotation retries (90,180,270)
            best_result = base_result
            best_angle = rotated_angle
            if base_result.get('confidence', 0.0) < 40:
                candidate_angles = []
                if angle in (90, 180, 270) and score >= MIN_ROTATE_TRY and score <= ROTATE_CONF_THRESHOLD:
                    candidate_angles.append(angle)
                for a in (90, 180, 270):
                    if a not in candidate_angles:
                        candidate_angles.append(a)

                for a in candidate_angles:
                    if a == 90:
                        trial_img = cv2.rotate(original_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    elif a == 180:
                        trial_img = cv2.rotate(original_img, cv2.ROTATE_180)
                    else:
                        trial_img = cv2.rotate(original_img, cv2.ROTATE_90_CLOCKWISE)

                    dn_trial = self.detect_and_normalize(trial_img)
                    trial_img2 = dn_trial.get('working_img', trial_img)
                    trial_boxes = dn_trial.get('boxes', []) or []
                    trial_det_ms = int(dn_trial.get('detection_time_ms', 0) or 0)
                    trial_res = self.recognize_from_boxes(trial_img2, trial_boxes, trial_det_ms)

                    prefer_trial = False
                    if trial_res.get('confidence', 0.0) > best_result.get('confidence', 0.0) + 5:
                        prefer_trial = True
                    elif trial_res.get('confidence', 0.0) > best_result.get('confidence', 0.0) and len(trial_res.get('text', '')) > len(best_result.get('text', '')):
                        prefer_trial = True

                    if prefer_trial:
                        best_result = trial_res
                        best_angle = a
                        logger.info(f"🔁 Rotation retry improved result with angle={a} (conf {trial_res.get('confidence')} vs {base_result.get('confidence')})")

            best_result['rotated_angle'] = int(best_angle or 0)
            return best_result

        except Exception as e:
            logger.error(f"Engine Error: {e}")
            return {"text_lines": [], "text": "", "confidence": 0.0, "rotated_angle": 0, "detection_time_ms": 0, "recognition_time_ms": 0, "boxes_count": 0, "valid_boxes_count": 0}
