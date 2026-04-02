import os
import re
import sys
import unicodedata
import joblib
from loguru import logger

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


def _normalize_text(text: str, keep_punct: bool = False) -> str:
    """Normalize text to ASCII-like uppercase for matching.

    - Removes combining diacritics, maps Đ/đ to D/d.
    - When `keep_punct` is True, preserves `/`, `\\`, `-`, `.` for pattern detection.
    """
    if text is None:
        return ""
    s = str(text)
    s = unicodedata.normalize('NFKD', s)
    # Map Vietnamese D-stroke explicitly (not decomposed by NFKD)
    s = s.replace('Đ', 'D').replace('đ', 'd')
    # Remove combining marks
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    s = s.upper()
    if keep_punct:
        s = re.sub(r'[^A-Z0-9\s/\\\-\.]', '', s)
    else:
        s = re.sub(r'[^A-Z0-9\s]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


# Precompute a normalized mapping for faster lookups in heuristics
NORMALIZED_DOC_TYPES = { _normalize_text(k): v for k, v in DOC_TYPES.items() }


class TextClassifier:
    def __init__(self, model_path: str = "models/phobert_classifier/model.pkl"):
        self.model_path = model_path
        self.is_mock = not os.path.exists(self.model_path)
        self.normalized_map = NORMALIZED_DOC_TYPES

        if self.is_mock:
            logger.warning("⚠️ Chưa có AI Model. Kích hoạt thuật toán Bóc tách Cấu trúc (Mock Mode)!")
        else:
            self.model = joblib.load(self.model_path)

    def _remove_accents(self, text: str) -> str:
        # Backwards-compatible helper: normalized text without punctuation
        return _normalize_text(text, keep_punct=False)

    def _normalize_keep_punct(self, text: str) -> str:
        return _normalize_text(text, keep_punct=True)

    def predict(self, text: str) -> dict:
        if not text or not text.strip():
            return {"label": "CHUA_PHAN_LOAI", "confidence": 0.0, "raw_title": ""}

        # If we have a trained model, prefer its prediction but map to standard labels
        if not self.is_mock:
            clean_text = text.replace('\n', ' ')
            predicted_label = self.model.predict([clean_text])[0]
            # If model returns already a standard label, use it
            if predicted_label in set(DOC_TYPES.values()):
                label = predicted_label
            else:
                # Try normalized lookup
                norm_pred = _normalize_text(predicted_label)
                label = self.normalized_map.get(norm_pred, DOC_TYPES.get(predicted_label, predicted_label))
            return {"label": label, "confidence": 0.90, "raw_title": "AI_PREDICTED"}

        lines = [line.strip() for line in text.split('\n') if len(line.strip()) > 3]
        text_norm_keep = self._normalize_keep_punct(text)
        text_norm = self._remove_accents(text)

        # Prefer explicit long-form keywords appearing anywhere in the page
        for keyword_norm, standard_label in self.normalized_map.items():
            if len(keyword_norm) > 3 and keyword_norm in text_norm:
                return {"label": standard_label, "confidence": 0.95, "raw_title": "Dựa trên từ khóa"}

        # 1. Bắt Ký hiệu Số (Chính xác tuyệt đối) - preserve slashes/hyphens
        for line in lines[:15]:
            norm_line = self._normalize_keep_punct(line)
            match = re.search(r'\b(SO|S)[:\s]*\d+[/\\\-]([A-Z0-9\-\.]+)', norm_line)
            if match:
                abbr = match.group(2).upper().split('-')[0].strip('.')
                if abbr in self.normalized_map:
                    return {"label": self.normalized_map[abbr], "confidence": 0.90, "raw_title": line}
                if abbr in DOC_TYPES:
                    return {"label": DOC_TYPES[abbr], "confidence": 0.90, "raw_title": line}

        # 2. Bắt Tên Loại Văn Bản (Strict Form) - per-line fallback
        for line in lines[:20]:
            clean_line = self._remove_accents(line)
            for keyword_norm, standard_label in self.normalized_map.items():
                if len(keyword_norm) > 3 and keyword_norm in clean_line:
                    return {"label": standard_label, "confidence": 0.85, "raw_title": line}

        # 3. Dấu hiệu Công văn (tolerate variations like V/V, V V, VV)
        if "KINH GUI" in text_norm or re.search(r'\bV[\/\\\.\s]?V\b', text_norm_keep):
            return {"label": "CONG_VAN", "confidence": 0.75, "raw_title": "Dựa trên form Công Văn"}

        return {"label": "KHAC", "confidence": 0.50, "raw_title": "Không xác định"}