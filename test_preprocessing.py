import os
import cv2
import json
from datetime import datetime

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

OCR_ENGINE = "tesseract"   # tesseract | paddle | trocr

SUPPORTED_EXT = (".jpg", ".jpeg", ".png", ".pdf")


# =====================================================
# MAIN
# =====================================================

def main():
    # 1️⃣ Pipeline config
    config = PreprocessingConfig(
        DPI=300,
        ENABLE_SCANNER_EFFECT=True,
        ENABLE_RED_STAMP=True,
        ENABLE_DESKEW=True,
        ENABLE_CROP=True,
        ENABLE_DENOISE=True,
        ENABLE_BINARIZATION=True
    )

    pipeline = PreprocessingPipeline(config)

    # 2️⃣ Chuẩn bị thư mục
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

    print(f"🚀 Tìm thấy {len(files)} file. OCR target = {OCR_ENGINE}")
    print("-" * 60)

    # 3️⃣ Chạy test
    for filename in files:
        file_path = os.path.join(INPUT_DIR, filename)
        print(f"\n📄 File: {filename}")

        for result in pipeline.run(file_path, ocr_engine=OCR_ENGINE):
            page_idx = result["page_index"]
            status = result["status"]

            if status != "success":
                print(f"❌ Page {page_idx} lỗi: {result.get('error')}")
                continue

            img = result["image"]
            meta = result["metadata"]

            base = os.path.splitext(filename)[0]
            img_name = f"{base}_p{page_idx}.jpg"
            meta_name = f"{base}_p{page_idx}.json"

            img_path = os.path.join(OUTPUT_DIR, img_name)
            meta_path = os.path.join(META_DIR, meta_name)

            # ⚠️ Đảm bảo img là uint8
            if img.dtype != "uint8":
                img = img.astype("uint8")

            cv2.imwrite(img_path, img)

            # Ghi metadata ra JSON
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

            # Log gọn gàng
            print(
                f"✅ P{page_idx} | "
                f"Type: {meta.get('doc_type')} | "
                f"Deskew: {meta.get('deskew_angle', 0):.2f}° | "
                f"RedStamp: {meta.get('has_red_stamp', False)} | "
                f"Time: {result['processing_time_ms']} ms"
            )

    print("\n🏁 TEST HOÀN TẤT")


# =====================================================
if __name__ == "__main__":
    main()
