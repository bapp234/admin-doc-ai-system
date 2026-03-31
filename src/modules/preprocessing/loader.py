from typing import Generator, Tuple
from pathlib import Path
import numpy as np
from loguru import logger
import cv2
from pdf2image import convert_from_path, pdfinfo_from_path
from pdf2image.exceptions import PDFInfoNotInstalledError, PDFSyntaxError

from PIL import Image, ImageOps # 🚀 Bổ sung module ImageOps của PIL

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    logger.warning("⚠️ Thư viện pillow-heif chưa được cài đặt.")

def load_image(file_path: str, dpi: int = 300, chunk_size: int = 10) -> Generator[Tuple[int, np.ndarray], None, None]:
    path = Path(file_path)
    if not path.exists(): 
        raise FileNotFoundError(f"404 Not Found: {file_path}")
    
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        try:
            info = pdfinfo_from_path(file_path)
            total_pages = info["Pages"]
            for start_page in range(1, total_pages + 1, chunk_size):
                end_page = min(start_page + chunk_size - 1, total_pages)
                try:
                    pages = convert_from_path(file_path, dpi=dpi, first_page=start_page, last_page=end_page, fmt="ppm", thread_count=2)
                    for i, page in enumerate(pages):
                        yield (start_page + i - 1), cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
                    del pages
                except Exception as e:
                    logger.warning(f"Chunk error {start_page}-{end_page}: {e}")
        except PDFInfoNotInstalledError:
            logger.critical("Thiếu Poppler! Vui lòng cài đặt poppler-utils.")
            raise
        except PDFSyntaxError:
            raise ValueError("PDF bị lỗi hoặc có mật khẩu.")

    elif suffix in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".heic"]:
        try:
            # 🚀 BÍ KÍP TRỊ LỖI CHỮ DỊ HỢM:
            # Đọc file bằng PIL và dùng ImageOps.exif_transpose để bẻ ảnh thẳng lại theo đúng góc chụp
            pil_img = Image.open(file_path)
            pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
            
            # Đẩy về dạng Numpy Array cho OpenCV xử lý ở các bước sau
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            if img is None: 
                raise ValueError(f"Không thể giải mã nội dung ảnh {suffix}.")
                
            yield 1, img
            
        except Exception as e:
            logger.error(f"Lỗi đọc ảnh {file_path}: {e}")
            raise 
    else:
        logger.warning(f"Định dạng không hỗ trợ: {suffix}")