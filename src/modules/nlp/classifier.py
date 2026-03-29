import os

class TextClassifier:
    def __init__(self, model_path: str = "models/phobert_classifier"):
        self.model_path = model_path
        self.is_mock = not os.path.exists(self.model_path)
        if not self.is_mock:
            pass

    def predict(self, text: str) -> dict:
        if not text.strip(): return {"label": "UNKNOWN", "confidence": 0.0}
        
        if not self.is_mock:
            pass

        text_upper = text.upper()
        if "QUYẾT ĐỊNH" in text_upper: return {"label": "QUYET_DINH", "confidence": 0.95}
        if "THÔNG BÁO" in text_upper: return {"label": "THONG_BAO", "confidence": 0.92}
        if "BÁO CÁO" in text_upper: return {"label": "BAO_CAO", "confidence": 0.88}
        if "BẰNG TỐT NGHIỆP" in text_upper: return {"label": "BANG_CAP", "confidence": 0.99}
        return {"label": "CONG_VAN", "confidence": 0.75}