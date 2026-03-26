import os
import csv
from loguru import logger

# Đường dẫn cấu hình chuẩn theo đúng project của bạn
OCR_DIR = "data/output/ocr_results"
DATASET_DIR = "data/dataset"
OUTPUT_CSV = os.path.join(DATASET_DIR, "text_classification_dataset.csv")

def auto_label(text: str) -> str:
    """
    Tự động gán nhãn dựa vào keyword để tạo Dataset mồi cho Luận văn 2.
    """
    text_upper = text.upper()
    if "QUYẾT ĐỊNH" in text_upper:
        return "QUYET_DINH"
    elif "THÔNG BÁO" in text_upper:
        return "THONG_BAO"
    elif "BÁO CÁO" in text_upper:
        return "BAO_CAO"
    elif "KÍNH GỬI" in text_upper or "V/V" in text_upper or "CỘNG HÒA" in text_upper:
        return "CONG_VAN"
    elif "BẰNG TỐT NGHIỆP" in text_upper or "CỬ NHÂN" in text_upper:
        return "BANG_CAP"
    else:
        return "KHAC"

def build():
    logger.info("⏳ Bắt đầu gom Text từ OCR để đóng gói Dataset Phân loại...")
    
    if not os.path.exists(OCR_DIR):
        logger.error(f"❌ Không tìm thấy thư mục OCR: {OCR_DIR}")
        return

    os.makedirs(DATASET_DIR, exist_ok=True)
    
    dataset = []
    # Quét tất cả file .txt trong ocr_results
    files = [f for f in os.listdir(OCR_DIR) if f.endswith('.txt')]
    
    if not files:
        logger.warning("⚠️ Không có file .txt nào để đóng gói!")
        return

    for filename in files:
        filepath = os.path.join(OCR_DIR, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
        if not content:
            continue
            
        # Loại bỏ các dấu xuống dòng để dữ liệu trên CSV nằm gọn trên 1 hàng
        clean_content = content.replace('\n', ' ') 
        
        # Phân loại tự động
        label = auto_label(clean_content)
        dataset.append([filename, clean_content, label])
        logger.info(f"📄 Đã đóng gói: {filename} -> Nhãn: {label}")

    # Ghi ra file CSV chuẩn mực
    with open(OUTPUT_CSV, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'text', 'label']) # Header các cột
        writer.writerows(dataset)
        
    logger.success(f"✅ Đã đóng gói thành công {len(dataset)} văn bản ra file: {OUTPUT_CSV}")
    logger.info("📦 HẠ TẦNG DATA ĐÃ SẴN SÀNG ĐỂ BÀN GIAO!")

if __name__ == "__main__":
    build()