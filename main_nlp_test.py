import os
import json
from loguru import logger
from src.modules.nlp.classifier import TextClassifier
from src.modules.nlp.extractor import InformationExtractor

OCR_RESULTS_DIR = "data/output/ocr_results"

def main():
    print("=" * 60)
    logger.info("Khởi động Mô-đun NLP (Classifier & Extractor)...")
    classifier = TextClassifier()
    extractor = InformationExtractor()
    print("=" * 60)

    files = [f for f in os.listdir(OCR_RESULTS_DIR) if f.endswith(".txt")]
    
    for filename in files:
        filepath = os.path.join(OCR_RESULTS_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        logger.info(f"📄 Đang xử lý NLP cho file: {filename}")
        
        # 1. Chạy Phân loại
        class_result = classifier.predict(text)
        
        # 2. Chạy Trích xuất
        extract_result = extractor.extract(text)

        # 3. Đóng gói kết quả JSON
        final_output = {
            "filename": filename,
            "classification": class_result,
            "extracted_info": extract_result
        }
        
        print(f"\n📊 KẾT QUẢ NLP: {filename}")
        print(json.dumps(final_output, indent=4, ensure_ascii=False))
        print("-" * 50)

if __name__ == "__main__":
    main()