import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
import gc
import torch
import cv2
import numpy as np
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

    def extract(self, img_array: np.ndarray) -> dict:
        try:
            working_img = img_array.copy()
            if len(working_img.shape) == 2:
                working_img = cv2.cvtColor(working_img, cv2.COLOR_GRAY2BGR)

            # 🚀 ĐEO "VÒNG KIM CÔ" CHO AI ĐOÁN GÓC:
            angle, score = self.detector.get_orientation(working_img)
            rotated_angle = 0
            
            # Chỉ cho phép lật ảnh nếu AI có độ tự tin cực cao (trên 95%)
            if score > 0.95:
                logger.info(f"🔄 AI phát hiện ảnh ngược {angle} độ (Độ tin cậy {score:.2f}). Đang lật lại...")
                rotated_angle = angle
                if angle == 90: working_img = cv2.rotate(working_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif angle == 180: working_img = cv2.rotate(working_img, cv2.ROTATE_180)
                elif angle == 270: working_img = cv2.rotate(working_img, cv2.ROTATE_90_CLOCKWISE)
            elif angle != 0:
                logger.debug(f"⚠️ Bỏ qua lật ảnh {angle} độ vì AI không đủ tự tin (Score: {score:.2f} < 0.95)")

            boxes = self.detector.detect(working_img)
            if not boxes: return {"text_lines": [], "text": "", "confidence": 0.0}

            valid_boxes = []
            cropped_images = []
            
            for box in boxes:
                pts = np.array(box, dtype=np.float32)
                w = int(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3])))
                h = int(max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2])))
                if w < 10 or h < 10 or h > w * 5: continue
                cropped_images.append(crop_and_rotate_box(working_img, box))
                valid_boxes.append(box)

            if not valid_boxes: return {"text_lines": [], "text": "", "confidence": 0.0}

            # Chạy OCR
            texts, probs = self.recognizer.recognize_batch(cropped_images)

            extracted_items = []
            for i, box in enumerate(valid_boxes):
                clean_text = clean_vba_text(texts[i])
                if clean_text:
                    cy = (box[0][1] + box[3][1]) / 2
                    left_x = min(box[0][0], box[3][0]) 
                    extracted_items.append({"text": clean_text, "cy": cy, "left_x": left_x, "prob": probs[i]})

            # 🚀 ĐÒN QUYẾT ĐỊNH: Dọn rác bộ nhớ (Tránh lỗi đọc ẩu ở các file sau)
            del cropped_images, working_img
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

            if not extracted_items: return {"text_lines": [], "text": "", "confidence": 0.0}

            extracted_items.sort(key=lambda item: item['cy'])
            lines = []
            current_line = [extracted_items[0]]
            
            for item in extracted_items[1:]:
                if abs(item['cy'] - current_line[-1]['cy']) < 15: current_line.append(item)
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

            return {
                "text_lines": final_text_lines, # 🚀 Trả về List để JSON xuống dòng đẹp mắt
                "text": "\n".join(final_text_lines),
                "confidence": round((total_prob / len(extracted_items)) * 100, 2),
                "rotated_angle": rotated_angle
            }

        except Exception as e:
            logger.error(f"Engine Error: {e}")
            return {"text_lines": [], "text": "", "confidence": 0.0, "rotated_angle": 0}