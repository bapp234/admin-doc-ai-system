import os
import torch
import cv2
import numpy as np
import logging
from loguru import logger

from src.modules.ocr.recognizer import TextRecognizer
from src.modules.ocr.layout import LayoutAnalyzer
from src.modules.ocr.cropper import crop_and_rotate_box
from src.modules.ocr.post_process import clean_vba_text

torch.set_num_threads(4)
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ['USERPROFILE'] = 'E:/admin-doc-ai-system'
os.environ['HOME'] = 'E:/admin-doc-ai-system'
logging.getLogger("ppocr").setLevel(logging.ERROR)

class OCREngine:
    def __init__(self, use_gpu: bool = False):
        self.recognizer = TextRecognizer(use_gpu=use_gpu)
        self.layout_analyzer = LayoutAnalyzer(use_gpu=use_gpu)

    def extract(self, img_array: np.ndarray) -> dict:
        try:
            working_img = img_array
            sorted_boxes = self.layout_analyzer.get_reading_order(working_img)
            rotated_angle = 0

            if len(sorted_boxes) < 3:
                test_img_ccw = cv2.rotate(img_array, cv2.ROTATE_90_COUNTERCLOCKWISE)
                boxes_ccw = self.layout_analyzer.get_reading_order(test_img_ccw)
                if len(boxes_ccw) > len(sorted_boxes):
                    working_img, sorted_boxes, rotated_angle = test_img_ccw, boxes_ccw, 270
                else:
                    test_img_cw = cv2.rotate(img_array, cv2.ROTATE_90_CLOCKWISE)
                    boxes_cw = self.layout_analyzer.get_reading_order(test_img_cw)
                    if len(boxes_cw) > len(sorted_boxes):
                        working_img, sorted_boxes, rotated_angle = test_img_cw, boxes_cw, 90

            if not sorted_boxes: 
                return {"text": "", "confidence": 0.0, "rotated_angle": 0}

            valid_boxes = [
                box for box in sorted_boxes
                if not (max(abs(box[0][0] - box[1][0]), abs(box[3][0] - box[2][0])) < 10 or 
                        max(abs(box[0][1] - box[3][1]), abs(box[1][1] - box[2][1])) < 10 or 
                        max(abs(box[0][1] - box[3][1]), abs(box[1][1] - box[2][1])) > max(abs(box[0][0] - box[1][0]), abs(box[3][0] - box[2][0])) * 3)
            ]

            extracted_items = []
            for box in valid_boxes:
                cropped_img = crop_and_rotate_box(working_img, box)
                text, prob = self.recognizer.recognize(cropped_img)
                clean_text = clean_vba_text(text)
                if clean_text:
                    cy = (box[0][1] + box[3][1]) / 2
                    left_x = min(box[0][0], box[3][0]) 
                    extracted_items.append({"text": clean_text, "cy": cy, "left_x": left_x, "prob": prob})

            if not extracted_items: 
                return {"text": "", "confidence": 0.0, "rotated_angle": rotated_angle}

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

            return {
                "text": "\n".join(final_text_lines),
                "confidence": round((total_prob / len(extracted_items)) * 100, 2),
                "rotated_angle": rotated_angle
            }

        except Exception as e:
            logger.error(f"Engine Error: {e}")
            return {"text": "", "confidence": 0.0, "rotated_angle": 0}