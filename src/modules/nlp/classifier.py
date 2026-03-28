import os
from loguru import logger

class TextClassifier:
    def __init__(self, model_path: str = "models/phobert_classifier"):
        """
        Khởi tạo mô hình Phân loại (Luận văn 2).
        """
        self.model_path = model_path
        self.is_mock = not os.path.exists(self.model_path)
        
        if self.is_mock:
            logger.warning(f"⚠️ Chưa tìm thấy Model Phân loại tại {model_path}. Chạy chế độ GIẢ LẬP (Mock Mode)!")
        else:
            logger.info(f"⏳ Đang tải PhoBERT Classifier từ {model_path}...")
            # TODO (Dành cho AI Trainer): Nạp mô hình HuggingFace tại đây
            # from transformers import pipeline
            # self.classifier = pipeline("text-classification", model=self.model_path)
            logger.info("✅ Mô hình Phân loại đã sẵn sàng!")

    def predict(self, text: str) -> dict:
        if not text.strip():
            return {"label": "UNKNOWN", "confidence": 0.0}

        if not self.is_mock:
            # TODO (Dành cho AI Trainer): Gọi hàm suy luận của mô hình thật
            # result = self.classifier(text[:256])[0]
            # return {"label": result['label'], "confidence": round(result['score'], 4)}
            pass

        # 🔴 LOGIC GIẢ LẬP (Mock Logic) để Backend test API
        text_upper = text.upper()
        if "QUYẾT ĐỊNH" in text_upper:
            return {"label": "QUYET_DINH", "confidence": 0.95}
        elif "THÔNG BÁO" in text_upper:
            return {"label": "THONG_BAO", "confidence": 0.92}
        elif "BÁO CÁO" in text_upper:
            return {"label": "BAO_CAO", "confidence": 0.88}
        elif "BẰNG TỐT NGHIỆP" in text_upper:
            return {"label": "BANG_CAP", "confidence": 0.99}
        else:
            return {"label": "CONG_VAN", "confidence": 0.75}