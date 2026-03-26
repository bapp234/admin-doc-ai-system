import os
import cv2
import json
from datetime import datetime

# Hãy đảm bảo thư mục của bạn tên là 'ocr', không phải 'orc' nhé!
from src.modules.ocr.engine import OCREngine
from src.modules.preprocessing.pipeline import (
    PreprocessingPipeline,
    PreprocessingConfig
)

# =====================================================
# CONFIG
# =====================================================

INPUT_DIR = "data/raw"
OUTPUT_DIR = "data/output"
META_DIR = "data/output/meta"

OCR_ENGINE = "tesseract"   # Chọn "paddle" nếu muốn AI xử lý mượt hơn

SUPPORTED_EXT = (".jpg", ".jpeg", ".png", ".pdf")


# =====================================================
# MAIN
# =====================================================

def main():
    # 1️⃣ Khởi tạo Cấu hình và Pipeline Tiền xử lý
    config = PreprocessingConfig(
        DPI=300,
        ENABLE_SCANNER_EFFECT=True,
        ENABLE_RED_STAMP=True, # Giữ lại dấu đỏ
        ENABLE_DESKEW=True,
        ENABLE_CROP=True,
        ENABLE_DENOISE=True,
        ENABLE_BINARIZATION=True
    )
    pipeline = PreprocessingPipeline(config)

    # 2️⃣ Khởi tạo Engine OCR
    print("-" * 60)
    print("⏳ Khởi tạo Engine OCR...")
    ocr = OCREngine()
    print("✅ Khởi tạo OCR thành công!")
    print("-" * 60)

    # 3️⃣ Chuẩn bị thư mục và đọc file
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(META_DIR, exist_ok=True)

    files = [
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(SUPPORTED_EXT)
    ]

    if not files:
        print(f"⚠️ Thư mục '{INPUT_DIR}' rỗng. Bỏ ảnh/PDF vào để test.")
        return

    print(f"🚀 Tìm thấy {len(files)} file. Sẵn sàng xử lý!")

    # 4️⃣ Chạy vòng lặp xử lý
    for filename in files:
        file_path = os.path.join(INPUT_DIR, filename)
        print(f"\n📄 Đang xử lý file: {filename}")

        for result in pipeline.run(file_path, ocr_engine=OCR_ENGINE):
            page_idx = result["page_index"]
            status = result["status"]

            if status != "success":
                print(f"❌ Page {page_idx} lỗi: {result.get('error')}")
                continue

            # Lấy ảnh và metadata
            img = result["image"]
            meta = result["metadata"]

            # --- PHẦN 1: LƯU ẢNH VÀ METADATA RA Ổ CỨNG ---
            base = os.path.splitext(filename)[0]
            img_path = os.path.join(OUTPUT_DIR, f"{base}_p{page_idx}.jpg")
            meta_path = os.path.join(META_DIR, f"{base}_p{page_idx}.json")

            if img.dtype != "uint8":
                img = img.astype("uint8")
            cv2.imwrite(img_path, img)

            meta_out = {
                "file": filename,
                "page_index": page_idx,
                "ocr_engine": OCR_ENGINE,
                "dpi": result["dpi"],
                "width": result["width"],
                "height": result["height"],
                "processing_time_ms": result["processing_time_ms"],
                "metadata": meta,
                "timestamp": datetime.now().isoformat()
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta_out, f, ensure_ascii=False, indent=2)

            # Log kết quả tiền xử lý
            print(
                f"✅ Tiền xử lý xong P{page_idx} | "
                f"Type: {meta.get('doc_type')} | "
                f"Deskew: {meta.get('deskew_angle', 0):.2f}° | "
                f"Time: {result['processing_time_ms']} ms"
            )

            # --- PHẦN 2: CHẠY OCR ĐỌC CHỮ ---
            print(f"🔄 Đang đọc chữ trên Trang {page_idx}...")
            
            # FIX LỖI: Truyền đúng biến img vào hàm extract
            ocr_result = ocr.extract(img)
                
            print(f"--- KẾT QUẢ OCR TRANG {page_idx} ---")
            
            # 🔴 SỬA LẠI 2 DÒNG NÀY CHO KHỚP VỚI DICT TRẢ VỀ TỪ ENGINE
            print(f"Độ tự tin trung bình: {ocr_result['confidence']}%")
            print("Nội dung:")
            print(ocr_result['text'])
            print("-" * 50)

    print("\n🏁 TEST HOÀN TẤT")

if __name__ == "__main__":
    main()