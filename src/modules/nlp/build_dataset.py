import sys
import os
import csv

OCR_DIR = "data/output/ocr_results"
DATASET_DIR = "data/dataset"
OUTPUT_CSV = os.path.join(DATASET_DIR, "text_classification_dataset.csv")

def auto_label(text: str) -> str:
    text_upper = text.upper()
    if "QUYẾT ĐỊNH" in text_upper: return "QUYET_DINH"
    if "THÔNG BÁO" in text_upper: return "THONG_BAO"
    if "BÁO CÁO" in text_upper: return "BAO_CAO"
    if any(k in text_upper for k in ["KÍNH GỬI", "V/V", "CỘNG HÒA"]): return "CONG_VAN"
    if any(k in text_upper for k in ["BẰNG TỐT NGHIỆP", "CỬ NHÂN"]): return "BANG_CAP"
    return "KHAC"

def build():
    if not os.path.exists(OCR_DIR): return
    os.makedirs(DATASET_DIR, exist_ok=True)
    
    dataset = []
    for filename in [f for f in os.listdir(OCR_DIR) if f.endswith('.txt')]:
        with open(os.path.join(OCR_DIR, filename), 'r', encoding='utf-8') as f:
            content = f.read().strip().replace('\n', ' ')
        if content:
            dataset.append([filename, content, auto_label(content)])

    if dataset:
        with open(OUTPUT_CSV, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['filename', 'text', 'label'])
            writer.writerows(dataset)

if __name__ == "__main__":
    build()