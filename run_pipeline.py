import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
os.environ['USERPROFILE'] = PROJECT_ROOT
os.environ['HOME'] = PROJECT_ROOT
import json
from loguru import logger
from src.modules.preprocessing.pipeline import PreprocessingPipeline
from src.modules.preprocessing.config import PreprocessingConfig
from src.modules.ocr.engine import OCREngine
from src.modules.nlp.classifier import TextClassifier
from src.modules.nlp.extractor import InformationExtractor

INPUT_DIR = "data/raw"
OUTPUT_DIR = "data/output/e2e_results"

def run_integration_test():
    print("=" * 60)
    logger.info("🚀 KHỞI ĐỘNG HỆ THỐNG KIỂM THỬ END-TO-END...")
    
    # 1. Khởi tạo toàn bộ các "Ổ cắm" AI
    prep_pipeline = PreprocessingPipeline(PreprocessingConfig())
    ocr_engine = OCREngine(use_gpu=False)
    classifier = TextClassifier()
    extractor = InformationExtractor()
    print("=" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf'))]
    
    if not files:
        logger.warning(f"⚠️ Thư mục {INPUT_DIR} đang trống! Hãy chép vài ảnh vào để test.")
        return

    for filename in files:
        file_path = os.path.join(INPUT_DIR, filename)
        logger.info(f"\n📦 ĐANG XỬ LÝ TÀI LIỆU: {filename}")
        
        # ---------------------------------------------------------
        # GIAI ĐOẠN 1: TIỀN XỬ LÝ (PREPROCESSING)
        # ---------------------------------------------------------
        logger.info("1️⃣ [PREPROCESSING] Đang căn chỉnh, khử nhiễu, xóa mộc đỏ...")
        for prep_result in prep_pipeline.run(file_path):
            img_processed = prep_result.get("image")
            page_index = prep_result.get("page_index", 1)
            prep_meta = prep_result.get("metadata", {})
            
            if img_processed is None:
                logger.error(f"❌ Tiền xử lý thất bại ở trang {page_index}")
                continue
                
            # ---------------------------------------------------------
            # GIAI ĐOẠN 2: NHẬN DẠNG QUANG HỌC (OCR)
            # ---------------------------------------------------------
            logger.info(f"2️⃣ [OCR] Đang bóc tách chữ và tái tạo Layout trang {page_index}...")
            ocr_result = ocr_engine.extract(img_processed)
            text_content = ocr_result.get("text", "")
            
            if not text_content:
                logger.warning(f"⚠️ Không tìm thấy chữ trên trang {page_index}.")
                continue
                
            # ---------------------------------------------------------
            # GIAI ĐOẠN 3: XỬ LÝ NGÔN NGỮ TỰ NHIÊN (NLP)
            # ---------------------------------------------------------
            logger.info("3️⃣ [NLP] Đang phân loại và trích xuất thực thể...")
            doc_class = classifier.predict(text_content)
            entities = extractor.extract(text_content)
            
            # ---------------------------------------------------------
            # GIAI ĐOẠN 4: ĐÓNG GÓI KẾT QUẢ VÀ XUẤT JSON
            # ---------------------------------------------------------
            final_output = {
                "file_info": {
                    "filename": filename,
                    "page": page_index
                },
                "preprocessing_logs": prep_meta,
                "ai_classification": doc_class,
                "extracted_data": entities,
                "ocr_confidence": ocr_result.get("confidence", 0.0),
                "full_text": text_content
            }
            
            json_filename = f"{os.path.splitext(filename)[0]}_page_{page_index}.json"
            json_path = os.path.join(OUTPUT_DIR, json_filename)
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(final_output, f, ensure_ascii=False, indent=4)
                
            logger.success(f"✅ Luồng dữ liệu THÔNG SUỐT! Kết quả lưu tại: {json_path}")

if __name__ == "__main__":
    run_integration_test()