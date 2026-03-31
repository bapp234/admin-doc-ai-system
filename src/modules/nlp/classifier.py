import os
import re
import sys
import unicodedata
import joblib
from loguru import logger

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

# CHUẨN HÓA 29 LOẠI VĂN BẢN (NĐ 30/2020)
DOC_TYPES = {
    "NGHI QUYET": "NGHI_QUYET", "NQ": "NGHI_QUYET",
    "QUYET DINH": "QUYET_DINH", "QD": "QUYET_DINH", "QĐ": "QUYET_DINH",
    "CHI THI": "CHI_THI", "CT": "CHI_THI",
    "QUY CHE": "QUY_CHE", "QC": "QUY_CHE",
    "QUY DINH": "QUY_DINH", "QYD": "QUY_DINH", "QYĐ": "QUY_DINH",
    "THONG CAO": "THONG_CAO", "TC": "THONG_CAO",
    "THONG BAO": "THONG_BAO", "TB": "THONG_BAO",
    "HUONG DAN": "HUONG_DAN", "HD": "HUONG_DAN",
    "CHUONG TRINH": "CHUONG_TRINH", "CTR": "CHUONG_TRINH",
    "KE HOACH": "KE_HOACH", "KH": "KE_HOACH",
    "PHUONG AN": "PHUONG_AN", "PA": "PHUONG_AN",
    "DE AN": "DE_AN", "ĐA": "DE_AN", "DA": "DE_AN",
    "BAO CAO": "BAO_CAO", "BC": "BAO_CAO",
    "BIEN BAN": "BIEN_BAN", "BB": "BIEN_BAN",
    "TO TRINH": "TO_TRINH", "TTR": "TO_TRINH",
    "HOP DONG": "HOP_DONG", "HĐ": "HOP_DONG",
    "CONG DIEN": "CONG_DIEN", "CĐ": "CONG_DIEN", "CD": "CONG_DIEN",
    "BAN GHI NHO": "BAN_GHI_NHO", "BGN": "BAN_GHI_NHO",
    "BAN THOA THUAN": "BAN_THOA_THUAN", "BTT": "BAN_THOA_THUAN",
    "GIAY UY QUYEN": "GIAY_UY_QUYEN", "GUQ": "GIAY_UY_QUYEN",
    "GIAY MOI": "GIAY_MOI", "GM": "GIAY_MOI",
    "GIAY GIOI THIEU": "GIAY_GIOI_THIEU", "GGT": "GIAY_GIOI_THIEU",
    "GIAY NGHI PHEP": "GIAY_NGHI_PHEP", "GNP": "GIAY_NGHI_PHEP",
    "PHIEU GUI": "PHIEU_GUI", "PG": "PHIEU_GUI",
    "PHIEU CHUYEN": "PHIEU_CHUYEN", "PC": "PHIEU_CHUYEN",
    "PHIEU BAO": "PHIEU_BAO", "PB": "PHIEU_BAO",
    "CONG VAN": "CONG_VAN", "CV": "CONG_VAN", "KINH GUI": "CONG_VAN"
}

class TextClassifier:
    def __init__(self, model_path: str = "models/phobert_classifier/model.pkl"):
        self.model_path = model_path
        self.is_mock = not os.path.exists(self.model_path)
        
        if self.is_mock:
            logger.warning("⚠️ Chưa có AI Model. Kích hoạt thuật toán Bóc tách Cấu trúc (Mock Mode)!")
        else:
            self.model = joblib.load(self.model_path)

    def _remove_accents(self, text: str) -> str:
        text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
        return re.sub(r'[^a-zA-Z0-9\s]', '', text).upper()

    def predict(self, text: str) -> dict:
        if not text.strip(): return {"label": "CHUA_PHAN_LOAI", "confidence": 0.0, "raw_title": ""}

        if not self.is_mock:
            clean_text = text.replace('\n', ' ')
            predicted_label = self.model.predict([clean_text])[0]
            # Ép kết quả AI về chuẩn 29 loại nếu nó lỡ dự đoán sai định dạng
            return {"label": DOC_TYPES.get(predicted_label, predicted_label), "confidence": 0.90, "raw_title": "AI_PREDICTED"}

        lines = [line.strip() for line in text.split('\n') if len(line.strip()) > 3]
        text_upper = self._remove_accents(text)

        # 1. Bắt Ký hiệu Số (Chính xác tuyệt đối)
        for line in lines[:15]:
            match = re.search(r'(SO|S)[:\s]*\d+[/\\\-](\w+)', self._remove_accents(line))
            if match:
                abbr = match.group(2).upper().split('-')[0]
                if abbr in DOC_TYPES:
                    return {"label": DOC_TYPES[abbr], "confidence": 0.95, "raw_title": line}

        # 2. Bắt Tên Loại Văn Bản (Strict Form)
        for line in lines[:20]:
            clean_line = self._remove_accents(line)
            for keyword, standard_label in DOC_TYPES.items():
                # Chỉ bắt các từ khóa dài rõ ràng, ví dụ "QUYET DINH"
                if len(keyword) > 3 and keyword in clean_line:
                    return {"label": standard_label, "confidence": 0.85, "raw_title": line}

        # 3. Dấu hiệu Công văn
        if "KINH GUI" in text_upper or "V/V" in text_upper:
            return {"label": "CONG_VAN", "confidence": 0.75, "raw_title": "Dựa trên form Công Văn"}

        return {"label": "KHAC", "confidence": 0.50, "raw_title": "Không xác định"}