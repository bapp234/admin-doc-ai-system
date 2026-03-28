import os
import re
from loguru import logger

class InformationExtractor:
    def __init__(self, model_path: str = "models/phobert_ner"):
        """
        Khởi tạo mô hình Trích xuất thông tin NER (Luận văn 3).
        """
        self.model_path = model_path
        self.is_mock = not os.path.exists(self.model_path)
        
        if self.is_mock:
            logger.warning(f"⚠️ Chưa tìm thấy Model NER tại {model_path}. Chạy chế độ GIẢ LẬP (Mock Mode)!")
        else:
            logger.info(f"⏳ Đang tải PhoBERT NER từ {model_path}...")
            # TODO (Dành cho AI Trainer): Nạp mô hình NER HuggingFace tại đây
            logger.info("✅ Mô hình NER đã sẵn sàng!")

    def extract(self, text: str) -> dict:
        """
        Đầu vào: Text OCR
        Đầu ra: Dictionary chứa các trường thông tin đã bóc tách
        """
        if not self.is_mock:
            # TODO (Dành cho AI Trainer): Gọi mô hình NER để bóc tách thực thể
            pass

        # 🔴 LOGIC GIẢ LẬP bằng Regex để Backend có data dựng giao diện
        extracted_data = {
            "so_hieu": None,
            "ngay_thang": None,
            "nguoi_ky": None,
            "ho_ten": None # Dành riêng cho Bằng cấp
        }

        # Tìm số hiệu (VD: Số: 4838/UBND-KTN hoặc Số hiệu T.02277211)
        so_hieu_match = re.search(r'(Số:|Số hiệu|Số vào sổ)[^\w]*([A-Z0-9/\.\-]+)', text, re.IGNORECASE)
        if so_hieu_match:
            extracted_data["so_hieu"] = so_hieu_match.group(2).strip()

        # Tìm ngày tháng (VD: ngày 28 tháng 12 năm 2022 hoặc 06/09/2004)
        date_match = re.search(r'(ngày\s+\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4}|\d{2}/\d{2}/\d{4})', text, re.IGNORECASE)
        if date_match:
            extracted_data["ngay_thang"] = date_match.group(1).strip()

        # Mock tìm Họ và tên (Nằm sau chữ "Họ và tên | ")
        name_match = re.search(r'Họ và tên\s*\|\s*([A-Z\sÀ-Ỹ]+)', text)
        if name_match:
            extracted_data["ho_ten"] = name_match.group(1).strip()

        return extracted_data