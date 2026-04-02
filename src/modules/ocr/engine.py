import sys
import os
import gc
import torch
import cv2
import numpy as np
import time
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

    def extract(self, img_array: np.ndarray, fast_mode: bool = False) -> dict:
        try:
            # Keep original copy for rotation trials
            original_img = img_array.copy()
            if len(original_img.shape) == 2:
                original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)

            # Quick orientation hint from detector (fast) — used as a candidate, not authoritative
            angle, score = self.detector.get_orientation(original_img)
            rotated_angle = 0

            # Strict rotate threshold is intentionally high; we'll still attempt guided retries below.
            ROTATE_CONF_THRESHOLD = 0.98
            MIN_ROTATE_TRY = 0.6

            def _detect(img):
                try:
                    det_start = time.time()
                    boxes = self.detector.detect(img)
                    det_end = time.time()
                    detection_time_ms = int((det_end - det_start) * 1000)
                    boxes_count = len(boxes) if boxes is not None else 0
                    return boxes, detection_time_ms, boxes_count
                except Exception as e:
                    logger.exception(f"Error during detection: {e}")
                    return None, 0, 0

            def _recognize_from_boxes(img, boxes, detection_time_ms):
                try:
                    boxes_count = len(boxes) if boxes is not None else 0
                    if not boxes:
                        return {"text_lines": [], "text": "", "confidence": 0.0, "detection_time_ms": detection_time_ms, "recognition_time_ms": 0, "boxes_count": boxes_count, "valid_boxes_count": 0}

                    valid_boxes = []
                    cropped_images = []
                    for box in boxes:
                        pts = np.array(box, dtype=np.float32)
                        w = int(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3])))
                        h = int(max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2])))
                        if w < 10 or h < 10 or h > w * 5:
                            continue
                        cropped_images.append(crop_and_rotate_box(img, box))
                        valid_boxes.append(box)

                    valid_boxes_count = len(valid_boxes)
                    logger.debug(f"🔍 Detection found {boxes_count} boxes, {valid_boxes_count} valid boxes")
                    if not valid_boxes:
                        return {"text_lines": [], "text": "", "confidence": 0.0, "detection_time_ms": detection_time_ms, "recognition_time_ms": 0, "boxes_count": boxes_count, "valid_boxes_count": valid_boxes_count}

                    rec_start = time.time()
                    texts, probs = self.recognizer.recognize_batch(cropped_images)
                    rec_end = time.time()
                    recognition_time_ms = int((rec_end - rec_start) * 1000)

                    extracted_items = []
                    for i, box in enumerate(valid_boxes):
                        clean_text = clean_vba_text(texts[i])
                        if clean_text:
                            cy = (box[0][1] + box[3][1]) / 2
                            left_x = min(box[0][0], box[3][0])
                            extracted_items.append({"text": clean_text, "cy": cy, "left_x": left_x, "prob": probs[i]})

                    # Memory hygiene
                    try:
                        del cropped_images
                    except Exception:
                        pass
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    if not extracted_items:
                        return {"text_lines": [], "text": "", "confidence": 0.0, "detection_time_ms": detection_time_ms, "recognition_time_ms": recognition_time_ms, "boxes_count": boxes_count, "valid_boxes_count": valid_boxes_count}

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
                        "detection_time_ms": detection_time_ms,
                        "recognition_time_ms": recognition_time_ms,
                        "boxes_count": boxes_count,
                        "valid_boxes_count": valid_boxes_count
                    }
                except Exception as e:
                    logger.exception(f"Error in recognize_from_boxes helper: {e}")
                    return {"text_lines": [], "text": "", "confidence": 0.0, "detection_time_ms": 0, "recognition_time_ms": 0, "boxes_count": 0, "valid_boxes_count": 0}

            # 1) Run on original orientation (or auto-rotated if very confident)
            if score > ROTATE_CONF_THRESHOLD and angle in (90, 180, 270):
                logger.info(f"🔄 AI phát hiện ảnh ngược {angle} độ (Độ tin cậy {score:.2f}). Đang lật lại...")
                if angle == 90:
                    working_img = cv2.rotate(original_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif angle == 180:
                    working_img = cv2.rotate(original_img, cv2.ROTATE_180)
                elif angle == 270:
                    working_img = cv2.rotate(original_img, cv2.ROTATE_90_CLOCKWISE)
                rotated_angle = angle
            else:
                working_img = original_img

            # Initial detection so we can do geometry-based orientation checks
            boxes, detection_time_ms, boxes_count = _detect(working_img)

            # Geometry-based landscape detection: count vertical vs horizontal boxes
            try:
                vertical_boxes = 0
                horizontal_boxes = 0
                if boxes:
                    for b in boxes:
                        pts = np.array(b, dtype=np.float32)
                        w = int(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3])))
                        h = int(max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2])))
                        if h > w * 1.2:
                            vertical_boxes += 1
                        elif w > h * 1.2:
                            horizontal_boxes += 1
                logger.debug(f"📐 Geometry check: vertical={vertical_boxes}, horizontal={horizontal_boxes}")

                # If vertical boxes dominate, the page is likely landscape - try rotating
                if vertical_boxes > horizontal_boxes:
                    logger.info("🧭 Geometry suggests LANDSCAPE (vertical boxes > horizontal). Trying rotate candidates to upright...")
                    candidates = [
                        (cv2.ROTATE_90_COUNTERCLOCKWISE, 90),
                        (cv2.ROTATE_90_CLOCKWISE, 270)
                    ]
                    found_upright = False
                    base_result = None
                    for rot_const, delta in candidates:
                        trial_img = cv2.rotate(working_img, rot_const)
                        trial_boxes, trial_det_ms, trial_boxes_count = _detect(trial_img)

                        # Count orientation in trial
                        t_vertical = 0
                        t_horizontal = 0
                        if trial_boxes:
                            for tb in trial_boxes:
                                pts = np.array(tb, dtype=np.float32)
                                w = int(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3])))
                                h = int(max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2])))
                                if h > w * 1.2:
                                    t_vertical += 1
                                elif w > h * 1.2:
                                    t_horizontal += 1

                        # If rotating produced horizontal-dominant boxes, accept it
                        if t_horizontal > t_vertical and trial_boxes:
                            logger.info(f"🔁 Geometry rotation {delta}° produced horizontal boxes -> selecting rotation.")
                            working_img = trial_img
                            boxes = trial_boxes
                            detection_time_ms = trial_det_ms
                            boxes_count = trial_boxes_count
                            rotated_angle = (rotated_angle + delta) % 360
                            # Recognize now that we've selected an upright candidate so base_result is always populated
                            base_result = _recognize_from_boxes(working_img, boxes, detection_time_ms)
                            found_upright = True
                            break

                        # Otherwise, do a quick recognition pass and accept if it improves confidence
                        trial_res = _recognize_from_boxes(trial_img, trial_boxes, trial_det_ms)
                        if trial_res.get('confidence', 0.0) > 50 and len(trial_res.get('text', '')) > len(("" if not boxes else "\n".join([str(x) for x in boxes]))):
                            logger.info(f"🔁 Geometry rotation {delta}° -> quick OCR indicates improvement (conf {trial_res.get('confidence')}). Selecting rotation.")
                            working_img = trial_img
                            boxes = trial_boxes
                            detection_time_ms = trial_det_ms
                            boxes_count = trial_boxes_count
                            rotated_angle = (rotated_angle + delta) % 360
                            base_result = trial_res
                            found_upright = True
                            break

                    if not found_upright:
                        # fall back to recognizing from the original working_img boxes
                        base_result = _recognize_from_boxes(working_img, boxes, detection_time_ms)
                else:
                    base_result = _recognize_from_boxes(working_img, boxes, detection_time_ms)
            except Exception as e:
                logger.exception(f"Geometry orientation check failed: {e}")
                base_result = _recognize_from_boxes(working_img, boxes, detection_time_ms)

            # Fast mode: return the base result immediately without rotation retries
            if fast_mode:
                base_result['rotated_angle'] = int(rotated_angle or 0)
                return base_result

            # Decide whether to attempt a guided rotation retry
            try_rotate = False
            # If detector suggested a rotation with moderate confidence, try that rotation as a candidate
            if angle in (90, 180, 270) and score >= MIN_ROTATE_TRY and score <= ROTATE_CONF_THRESHOLD:
                try_rotate = True
                candidate_angles = [angle]
            else:
                candidate_angles = []

            # Also consider trying rotations if OCR confidence is very low
            if base_result.get('confidence', 0.0) < 40:
                # try all three rotations (but avoid duplicating detector-suggested angle)
                for a in (90, 180, 270):
                    if a not in candidate_angles:
                        candidate_angles.append(a)
                try_rotate = True

            best_result = base_result
            best_angle = rotated_angle

            if try_rotate and candidate_angles:
                for a in candidate_angles:
                    # Rotate original image by 'a' and evaluate
                    if a == 90:
                        trial_img = cv2.rotate(original_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    elif a == 180:
                        trial_img = cv2.rotate(original_img, cv2.ROTATE_180)
                    else:
                        trial_img = cv2.rotate(original_img, cv2.ROTATE_90_CLOCKWISE)

                    trial_boxes, trial_det_ms, trial_boxes_count = _detect(trial_img)
                    trial_res = _recognize_from_boxes(trial_img, trial_boxes, trial_det_ms)
                    # prefer higher confidence, then longer text
                    prefer_trial = False
                    if trial_res.get('confidence', 0.0) > best_result.get('confidence', 0.0) + 5:
                        prefer_trial = True
                    elif trial_res.get('confidence', 0.0) > best_result.get('confidence', 0.0) and len(trial_res.get('text', '')) > len(best_result.get('text', '')):
                        prefer_trial = True

                    if prefer_trial:
                        best_result = trial_res
                        best_angle = a
                        logger.info(f"🔁 Rotation retry improved result with angle={a} (conf {trial_res.get('confidence')} vs {base_result.get('confidence')})")

            # Attach chosen rotation angle if any
            best_result['rotated_angle'] = int(best_angle or 0)
            return best_result

        except Exception as e:
            logger.error(f"Engine Error: {e}")
            return {"text_lines": [], "text": "", "confidence": 0.0, "rotated_angle": 0, "detection_time_ms": 0, "recognition_time_ms": 0, "boxes_count": 0, "valid_boxes_count": 0}