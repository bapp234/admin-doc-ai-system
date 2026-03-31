import os
import cv2
import json
import csv
from loguru import logger

from src.modules.preprocessing.pipeline import PreprocessingPipeline
from src.modules.preprocessing.config import PreprocessingConfig
from src.modules.ocr.engine import OCREngine
from src.modules.nlp.classifier import TextClassifier
from src.modules.nlp.extractor import InformationExtractor
from src.utils.hardware import optimize_hardware

INPUT_DIR = "data/raw"
OUTPUT_BASE_DIR = "data/output/classified_docs"
DATASET_CSV = "data/dataset/training_dataset.csv"

def init_dataset_file():
    os.makedirs(os.path.dirname(DATASET_CSV), exist_ok=True)
    if not os.path.exists(DATASET_CSV):
        with open(DATASET_CSV, 'w', encoding='utf-8-sig', newline='') as f:
            csv.writer(f).writerow(['filename', 'label', 'raw_title', 'clean_text'])

def main():
    print("=" * 60)
    logger.info("🚀 KHỞI ĐỘNG HỆ THỐNG V3 (MULTI-PAGE PDF & TRACEABILITY)")
    
    use_gpu = optimize_hardware()
    init_dataset_file()
    
    prep_config = PreprocessingConfig()
    prep_pipeline = PreprocessingPipeline(prep_config)
    
    ocr_engine = OCREngine(use_gpu=use_gpu)
    classifier = TextClassifier()
    extractor = InformationExtractor()
    
    valid_exts = {'.png', '.jpg', '.jpeg', '.pdf', '.heic'}
    files = [f for f in os.listdir(INPUT_DIR) if os.path.splitext(f)[1].lower() in valid_exts]
    
    if not files:
        logger.warning(f"⚠️ Thư mục {INPUT_DIR} trống.")
        return

    with open(DATASET_CSV, 'a', encoding='utf-8-sig', newline='') as dataset_file:
        csv_writer = csv.writer(dataset_file)
        
        for filename in files:
            file_path = os.path.join(INPUT_DIR, filename)
            base_name = os.path.splitext(filename)[0]
            logger.info(f"\n📦 ĐANG XỬ LÝ: {filename}")
            
            for prep_result in prep_pipeline.run(file_path, ocr_engine="paddle"):
                img_processed = prep_result.get("image")
                page_index = prep_result.get("page_index", 1) # 🚀 LẤY SỐ TRANG
                
                if img_processed is None: continue
                    
                ocr_result = ocr_engine.extract(img_processed)
                text_content = ocr_result.get("text", "")
                text_lines = ocr_result.get("text_lines", [])
                
                if not text_content: continue
                    
                doc_class = classifier.predict(text_content)
                entities = extractor.extract(text_content)
                
                label = doc_class.get("label", "CHUA_PHAN_LOAI")
                raw_title = doc_class.get("raw_title", "")
                
                # =========================================================
                # LOGIC LƯU TRỮ CHUYÊN SÂU (MULTI-PAGE)
                # =========================================================
                target_folder = os.path.join(OUTPUT_BASE_DIR, label)
                os.makedirs(target_folder, exist_ok=True)
                
                # 1. Định danh cấp độ trang (VD: hoso_page_1)
                page_identifier = f"{base_name}_page_{page_index}"
                
                # 2. Lưu trực tiếp MỘT TRANG ĐÓ thành ảnh JPG thay vì copy cả cục PDF
                target_image_path = os.path.join(target_folder, f"{page_identifier}.jpg")
                cv2.imwrite(target_image_path, img_processed)
                
                # 3. Format JSON có tính năng LƯU VẾT (Traceability)
                final_data = {
                    "original_file": filename,   # 🚀 Phục vụ việc gom file sau này
                    "page_index": page_index,    # 🚀 Thứ tự trang
                    "document_type": label,
                    "confidence": doc_class.get("confidence", 0.0),
                    "extracted_entities": entities,
                    "clean_text_lines": text_lines 
                }
                
                json_path = os.path.join(target_folder, f"{page_identifier}.json")
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(final_data, f, ensure_ascii=False, indent=4)
                    
                # 4. Ghi rõ số trang vào Dataset
                csv_writer.writerow([f"{page_identifier}.jpg", label, raw_title, text_content.replace('\n', ' ')])
                
                logger.success(f"✅ Label [{label}] -> {page_identifier} -> OK")

if __name__ == "__main__":
    main()