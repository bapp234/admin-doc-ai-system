import os
import logging

# --- KHÓA CHẶT BỘ TĂNG TỐC LỖI CỦA C++ ---
os.environ["FLAGS_use_mkldnn"] = "0"  

# Đánh lừa đường dẫn Tiếng Việt (Giữ nguyên như cũ)
os.environ['USERPROFILE'] = 'E:/admin-doc-ai-system'
os.environ['HOME'] = 'E:/admin-doc-ai-system'

import numpy as np
import gc
from loguru import logger
from paddleocr import PaddleOCR

# Ép hệ thống log của PaddleOCR (ppocr) chỉ hiển thị lỗi
logging.getLogger("ppocr").setLevel(logging.ERROR)

class OCREngine:
    def __init__(self):
        """Khởi tạo Engine OCR một lần duy nhất để tối ưu RAM/VRAM."""
        logger.info("⏳ Đang nạp mô hình PaddleOCR (Bản 2.8.1 + Paddle 2.6.2)...")
        self.model = PaddleOCR(
            use_angle_cls=True, 
            lang="vi", 
            use_gpu=False,
            show_log=False,
            enable_mkldnn=False # Cứ để lại cho chắc ăn
        )
        logger.info("✅ Nạp PaddleOCR thành công!")

    def _sort_boxes(self, ocr_result: list, y_tolerance: int = 10) -> list:
        """
        Thuật toán cực kỳ quan trọng: Sắp xếp chữ từ trên xuống, từ trái qua phải.
        y_tolerance: Sai số pixel trục Y để coi 2 chữ là nằm trên cùng 1 dòng.
        """
        if not ocr_result:
            return []

        # Cấu trúc 1 item trong ocr_result: [box_coords, (text, confidence)]
        # box_coords có 4 điểm: top_left, top_right, bottom_right, bottom_left
        
        # 1. Sắp xếp sơ bộ theo trục Y (từ trên xuống dưới)
        sorted_by_y = sorted(ocr_result, key=lambda x: x[0][0][1])

        # 2. Gom nhóm các chữ trên cùng 1 dòng và sắp xếp theo trục X (trái qua phải)
        lines = []
        current_line = []
        
        for item in sorted_by_y:
            if not current_line:
                current_line.append(item)
                continue
            
            # Lấy Y của chữ hiện tại so sánh với Y của chữ đầu tiên trong dòng
            current_y = item[0][0][1]
            base_y = current_line[0][0][0][1]
            
            if abs(current_y - base_y) <= y_tolerance:
                # Nằm trên cùng 1 dòng -> Thêm vào dòng hiện tại
                current_line.append(item)
            else:
                # Khác dòng -> Chốt dòng cũ (sort theo X), tạo dòng mới
                current_line.sort(key=lambda x: x[0][0][0])
                lines.extend(current_line)
                current_line = [item]
                
        # Nhớ chốt dòng cuối cùng
        if current_line:
            current_line.sort(key=lambda x: x[0][0][0])
            lines.extend(current_line)

        return lines

    def extract(self, image: np.ndarray) -> dict:
        """
        Thực thi OCR trên một ma trận ảnh.
        Trả về Dictionary chứa văn bản đầy đủ và dữ liệu chi tiết.
        """
        # 1. PHÒNG THỦ: Kiểm tra ảnh đầu vào
        if image is None or image.size == 0:
            logger.warning("Ảnh đầu vào rỗng hoặc lỗi.")
            return {"full_text": "", "details": [], "mean_confidence": 0.0}

        try:
            # 2. CHẠY AI
            # cls=True để kích hoạt mô hình nhận diện góc nghiêng (lật ngược ảnh)
            result = self.model.ocr(image, cls=True)
            
            # Paddle trả về list lồng nhau. Nếu ảnh không có chữ, result[0] sẽ là None.
            if not result or result[0] is None:
                return {"full_text": "", "details": [], "mean_confidence": 0.0}

            raw_data = result[0]

            # 3. HẬU XỬ LÝ: Sắp xếp lại tọa độ
            sorted_data = self._sort_boxes(raw_data)

            # 4. TỔNG HỢP KẾT QUẢ
            extracted_lines = []
            details = []
            total_conf = 0.0

            for line in sorted_data:
                box = line[0]
                text = line[1][0]
                conf = float(line[1][1])

                extracted_lines.append(text)
                total_conf += conf
                
                details.append({
                    "text": text,
                    "confidence": round(conf, 4),
                    "box": box # Tọa độ [top_left, top_right, bottom_right, bottom_left]
                })

            full_text = "\n".join(extracted_lines)
            mean_conf = total_conf / len(sorted_data) if sorted_data else 0.0

            return {
                "full_text": full_text,
                "mean_confidence": round(mean_conf, 4), # Dùng để đánh giá AI đọc tốt hay tệ
                "details": details # Dùng để nhét vào PhoBERT hoặc vẽ khung đỏ sau này
            }

        except Exception as e:
            logger.error(f"Lỗi khi chạy PaddleOCR: {e}")
            return {"full_text": "", "details": [], "mean_confidence": 0.0}
            
        finally:
            # Luôn dọn dẹp RAM dù có lỗi hay không
            gc.collect()