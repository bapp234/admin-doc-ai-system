import os
import cv2
import time # 🔴 Thêm thư viện thời gian
from loguru import logger
from src.modules.ocr.engine import OCREngine

INPUT_DIR = "data/raw"
OUTPUT_DIR = "data/output/ocr_results"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 60)
    logger.info("⏳ Khởi tạo hệ thống OCR Hybrid (docTR + VietOCR + 2D Layout)...")
    engine = OCREngine(use_gpu=False)
    print("=" * 60)
    
    valid_extensions = [".jpg", ".jpeg", ".png"]
    files = [f for f in os.listdir(INPUT_DIR) if os.path.splitext(f)[1].lower() in valid_extensions]
    
    if not files:
        logger.warning(f"⚠️ Không tìm thấy ảnh nào trong {INPUT_DIR}")
        return

    logger.info(f"🚀 Tìm thấy {len(files)} file. Sẵn sàng test!")
    
    total_start_time = time.time() # Bắt đầu bấm giờ tổng

    for filename in files:
        img_path = os.path.join(INPUT_DIR, filename)
        logger.info(f"📄 Đang xử lý: {filename} ...")
        
        img = cv2.imread(img_path)
        if img is None: continue
            
        file_start_time = time.time() # Bấm giờ từng file
        result = engine.extract(img)
        file_end_time = time.time()
        
        process_time = round(file_end_time - file_start_time, 2)
        
        print(f"\n{'='*50}")
        print(f"📊 KẾT QUẢ FILE: {filename} (⏱️ Thời gian: {process_time}s)")
        print(f"📐 Góc xoay tự động nắn: {result.get('rotated_angle', 0)} độ")
        print(f"🎯 Độ tự tin trung bình: {result.get('confidence', 0)}%")
        print("-" * 50)
        print("NỘI DUNG ĐÃ ĐỌC:")
        print(result.get('text', ''))
        print(f"{'='*50}\n")
        
        out_txt_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(filename)[0]}.txt")
        with open(out_txt_path, "w", encoding="utf-8") as f:
            f.write(result.get('text', ''))
            
    total_time = round(time.time() - total_start_time, 2)
    logger.success(f"✅ Hoàn tất! Tổng thời gian chạy: {total_time} giây. Text đã xuất ra: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()