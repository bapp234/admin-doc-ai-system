import sys
import os
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

        # First, normalize likely document-number tokens in the text to reduce OCR confusions
        cleaned_text = self.normalize_document_numbers(text)

        extracted_data = {k: None for k in ["so_hieu", "ngay_thang", "nguoi_ky", "ho_ten"]}

        # Try common patterns for document number (so_hieu) using cleaned text
        so_hieu_match = re.search(r'(Số:|Số hiệu|Số vào sổ)[^\w]*([A-Z0-9/\.\-]+)', cleaned_text, re.IGNORECASE)
        if so_hieu_match:
            extracted_data["so_hieu"] = so_hieu_match.group(2).strip()

        date_match = re.search(r'(ngày\s+\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4}|\d{2}/\d{2}/\d{4})', cleaned_text, re.IGNORECASE)
        if date_match:
            extracted_data["ngay_thang"] = date_match.group(1).strip()

        name_match = re.search(r'Họ và tên\s*\|\s*([A-Z\sÀ-Ỹ]+)', cleaned_text)
        if name_match:
            extracted_data["ho_ten"] = name_match.group(1).strip()

        return extracted_data

    def normalize_document_numbers(self, text: str) -> str:
        """Normalize common OCR confusions inside tokens that likely represent document numbers.

        Replacements are conservative: only tokens that contain at least one digit are normalized
        so we don't mangle normal words. This fixes O->0, I/l/|->1, S->5, etc inside IDs.
        """
        if not text:
            return text

        ambig_map = {
            'O': '0', 'o': '0',
            'I': '1', 'l': '1', '|': '1', 'i': '1',
            'S': '5', 's': '5',
            'B': '8',
            'Z': '2',
            'Q': '0'
        }

        def _norm_token(m):
            tok = m.group(0)
            # Only normalize tokens that contain digits or obvious mixed alnum patterns
            if re.search(r'\d', tok) or re.search(r'[A-Z]{1,}[0-9]{1,}', tok, re.IGNORECASE):
                return ''.join(ambig_map.get(ch, ch) for ch in tok)
            return tok

        # Replace only alnum/separator tokens (keep diacritics and Vietnamese words untouched)
        return re.sub(r'[A-Za-z0-9\-/\.\|]+', _norm_token, text)