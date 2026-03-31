import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
import re

class InformationExtractor:
    def __init__(self, model_path: str = "models/phobert_ner"):
        self.model_path = model_path
        self.is_mock = not os.path.exists(self.model_path)
        if not self.is_mock:
            pass

    def extract(self, text: str) -> dict:
        if not self.is_mock:
            pass

        extracted_data = {k: None for k in ["so_hieu", "ngay_thang", "nguoi_ky", "ho_ten"]}

        so_hieu_match = re.search(r'(Số:|Số hiệu|Số vào sổ)[^\w]*([A-Z0-9/\.\-]+)', text, re.IGNORECASE)
        if so_hieu_match: extracted_data["so_hieu"] = so_hieu_match.group(2).strip()

        date_match = re.search(r'(ngày\s+\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4}|\d{2}/\d{2}/\d{4})', text, re.IGNORECASE)
        if date_match: extracted_data["ngay_thang"] = date_match.group(1).strip()

        name_match = re.search(r'Họ và tên\s*\|\s*([A-Z\sÀ-Ỹ]+)', text)
        if name_match: extracted_data["ho_ten"] = name_match.group(1).strip()

        return extracted_data