import os
import torch
import cv2
import numpy as np
import logging
from loguru import logger

from src.modules.ocr.detector import TextDetector
from src.modules.ocr.recognizer import TextRecognizer
from src.modules.ocr.cropper import crop_and_rotate_box
from src.modules.ocr.post_process import clean_vba_text

# Tối ưu hóa CPU/GPU cho môi trường AI
torch.set_num_threads(4)
os.environ["FLAGS_use_mkldnn"] = "0"
logging.getLogger("ppocr").setLevel(logging.ERROR)

class OCREngine:
    def __init__(self, use_gpu: bool = False):
        self.detector = TextDetector(use_gpu=use_gpu)
        self.recognizer = TextRecognizer(use_gpu=use_gpu)

    def extract(self, img_array: np.ndarray) -> dict:
        try:
            working_img = img_array
            
            # 1. Vá lỗi kênh màu (Đảm bảo ảnh luôn là 3 chiều - BGR)
            if len(working_img.shape) == 2:
                working_img = cv2.cvtColor(working_img, cv2.COLOR_GRAY2BGR)

            # 2. Tự động xoay ảnh thông minh bằng AI (Thay vì brute-force)
            angle, _ = self.detector.get_orientation(working_img)
            rotated_angle = angle
            if angle == 90:
                working_img = cv2.rotate(working_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif angle == 180:
                working_img = cv2.rotate(working_img, cv2.ROTATE_180)
            elif angle == 270:
                working_img = cv2.rotate(working_img, cv2.ROTATE_90_CLOCKWISE)

            # 3. Detect tọa độ chữ (Box)
            boxes = self.detector.detect(working_img)
            if not boxes: 
                return {"text": "", "confidence": 0.0, "rotated_angle": rotated_angle}

            # 4. Trích xuất, cắt và đọc chữ
            extracted_items = []
            for box in boxes:
                # Lọc box nhiễu/rác quá nhỏ hoặc dị dạng
                pts = np.array(box, dtype=np.float32)
                w = int(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3])))
                h = int(max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2])))
                if w < 10 or h < 10 or h > w * 5:
                    continue

                cropped_img = crop_and_rotate_box(working_img, box)
                text, prob = self.recognizer.recognize(cropped_img)
                clean_text = clean_vba_text(text)
                
                if clean_text:
                    cy = (box[0][1] + box[3][1]) / 2
                    left_x = min(box[0][0], box[3][0]) 
                    extracted_items.append({"text": clean_text, "cy": cy, "left_x": left_x, "prob": prob})

            if not extracted_items: 
                return {"text": "", "confidence": 0.0, "rotated_angle": rotated_angle}

            # 5. Khôi phục Bố cục (Layout Reconstruction)
            extracted_items.sort(key=lambda item: item['cy'])
            lines = []
            current_line = [extracted_items[0]]
            
            for item in extracted_items[1:]:
                # Gom các box nằm trên cùng 1 hàng ngang (dung sai 15 pixel)
                if abs(item['cy'] - current_line[-1]['cy']) < 15:
                    current_line.append(item)
                else:
                    lines.append(current_line)
                    current_line = [item]
            lines.append(current_line)

            # 6. Sắp xếp lại theo cột (trục X) và nối chuỗi
            final_text_lines = []
            total_prob = 0
            
            for line in lines:
                line.sort(key=lambda item: item['left_x'])
                line_text = " | ".join([item['text'] for item in line])
                final_text_lines.append(line_text)
                total_prob += sum([item['prob'] for item in line])

            return {
                "text": "\n".join(final_text_lines),
                "confidence": round((total_prob / len(extracted_items)) * 100, 2),
                "rotated_angle": rotated_angle
            }

        except Exception as e:
            logger.error(f"Engine Error: {e}")
            return {"text": "", "confidence": 0.0, "rotated_angle": 0}