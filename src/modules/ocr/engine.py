import torch
import os

# Ép PyTorch chỉ dùng 4 luồng CPU thay vì chiếm dụng toàn bộ, giúp máy không bị giật lag
torch.set_num_threads(4) 

os.environ["FLAGS_use_mkldnn"] = "0"
os.environ['USERPROFILE'] = 'E:/admin-doc-ai-system'
os.environ['HOME'] = 'E:/admin-doc-ai-system'

import cv2
import numpy as np
import logging
from loguru import logger

from src.modules.ocr.recognizer import TextRecognizer
from src.modules.ocr.layout import LayoutAnalyzer
from src.modules.ocr.cropper import crop_and_rotate_box
from src.modules.ocr.post_process import clean_vba_text

logging.getLogger("ppocr").setLevel(logging.ERROR)

class OCREngine:
    def __init__(self, use_gpu: bool = False):
        # use_gpu = False vì máy chúng ta dùng Intel UHD, không có CUDA
        logger.info("⏳ Đang nạp Hệ thống Nhận dạng (Optimized CPU Mode)...")
        self.recognizer = TextRecognizer(use_gpu=False)
        self.layout_analyzer = LayoutAnalyzer(use_gpu=False)
        logger.info("✅ Engine Modular đã sẵn sàng chiến đấu!")

    def extract(self, img_array: np.ndarray) -> dict:
        try:
            working_img = img_array
            sorted_boxes = self.layout_analyzer.get_reading_order(working_img)
            rotated_angle = 0

            # THUẬT TOÁN THỬ SAI nắn ảnh dọc (Giữ nguyên)
            if len(sorted_boxes) < 3:
                logger.warning("⚠️ docTR tìm thấy quá ít chữ. Đang thử xoay ảnh...")
                test_img_ccw = cv2.rotate(img_array, cv2.ROTATE_90_COUNTERCLOCKWISE)
                boxes_ccw = self.layout_analyzer.get_reading_order(test_img_ccw)
                if len(boxes_ccw) > len(sorted_boxes):
                    working_img, sorted_boxes, rotated_angle = test_img_ccw, boxes_ccw, 270
                    logger.success("✅ Nắn dọc ảnh thành công (270 độ)!")
                else:
                    test_img_cw = cv2.rotate(img_array, cv2.ROTATE_90_CLOCKWISE)
                    boxes_cw = self.layout_analyzer.get_reading_order(test_img_cw)
                    if len(boxes_cw) > len(sorted_boxes):
                        working_img, sorted_boxes, rotated_angle = test_img_cw, boxes_cw, 90
                        logger.success("✅ Nắn dọc ảnh thành công (90 độ)!")

            if not sorted_boxes: return {"text": "", "confidence": 0.0, "rotated_angle": 0}

            # LỌC HÌNH HỌC (Giữ nguyên)
            valid_boxes = []
            for box in sorted_boxes:
                w = max(abs(box[0][0] - box[1][0]), abs(box[3][0] - box[2][0]))
                h = max(abs(box[0][1] - box[3][1]), abs(box[1][1] - box[2][1]))
                if w < 10 or h < 10 or h > w * 3: continue 
                valid_boxes.append(box)

            # 🔴 LOGIC MỚI: TÁI TẠO BỐ CỤC 2D (Chống nhầm cột)
            extracted_items = []
            for box in valid_boxes:
                mảnh_ảnh = crop_and_rotate_box(working_img, box)
                text, prob = self.recognizer.recognize(mảnh_ảnh)
                clean_text = clean_vba_text(text)
                
                if clean_text:
                    cy=(box[0][1]+box[3][1])/2
                    left_x=(box[0][0]+box[3][0])
                    extracted_items.append({"text": clean_text, "cy": cy, "left_x": left_x, "prob": prob})
            if not extracted_items: return {"text": "", "confidence": 0.0, "rotated_angle": rotated_angle}

            # Sắp xếp toàn bộ chữ từ trên xuống dưới theo trục Y
            extracted_items.sort(key=lambda item: item['cy'])
            lines = []
            current_line = [extracted_items[0]]
            y_threshold = 15

            for item in extracted_items[1:]:
                if abs(item['cy'] - current_line[-1]['cy']) < y_threshold:
                    current_line.append(item)
                else:
                    lines.append(current_line)
                    current_line = [item]
            lines.append(current_line)

            final_text_lines = []
            total_prob = 0
            
            for line in lines:
                # Sửa ở đây: Sắp xếp các chữ trong cùng hàng ngang theo Mép trái
                line.sort(key=lambda item: item['left_x'])
                line_text = " | ".join([item['text'] for item in line])
                final_text_lines.append(line_text)
                total_prob += sum([item['prob'] for item in line])

            avg_conf = (total_prob / len(extracted_items)) * 100
            
            return {
                "text": "\n".join(final_text_lines),
                "confidence": round(avg_conf, 2),
                "rotated_angle": rotated_angle
            }

        except Exception as e:
            logger.error(f"❌ Lỗi Engine: {e}")
            return {"text": f"Error: {str(e)}", "confidence": 0.0, "rotated_angle": 0}