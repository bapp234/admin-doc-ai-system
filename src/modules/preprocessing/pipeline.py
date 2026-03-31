import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
import time
import cv2
import numpy as np
from typing import Dict, Any, Generator
from loguru import logger

from .config import PreprocessingConfig
from .profiles import apply_profile
from .normalize import normalize_for_ocr
from .validators import is_too_blurry, is_blank
from .loader import load_image
from .detector import detect_document_type
from .alignment import auto_scan_document
from .deskew import deskew_image
from .crop import crop
from .filters import denoise, binarize, remove_shadows
from .stamp import red_stamp


class PreprocessingPipeline:

    def __init__(self, config: PreprocessingConfig | None = None):
        self.config = config or PreprocessingConfig()

    def run(self, file_path: str, ocr_engine: str = "tesseract"
            ) -> Generator[Dict[str, Any], None, None]:

        logger.info(f"🚀 Preprocessing start: {file_path}")
        start = time.time()

        for page_index, img in load_image(file_path, dpi=self.config.DPI):
            try:
                result = self.process(img, page_index, ocr_engine)
                yield result
                del img
            except Exception as e:
                logger.error(f"❌ Page {page_index}: {e}")
                yield {
                    "page_index": page_index,
                    "status": "failed",
                    "error": str(e)
                }

        logger.success(
            f"🏁 Done in {int((time.time() - start)*1000)} ms"
        )

    def process(self, img: np.ndarray, page_index: int, ocr_engine: str
                ) -> Dict[str, Any]:

        t0 = time.time()
        metadata = {}

        doc_type = detect_document_type(img)
        metadata["doc_type"] = doc_type

        apply_profile(self.config, doc_type)

        h, w = img.shape[:2]
        if max(h, w) > self.config.MAX_IMAGE_SIZE:
            scale = self.config.MAX_IMAGE_SIZE / max(h, w)
            img = cv2.resize(img, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_AREA)
            metadata["resized"] = True

        if doc_type == "camera":
            img = remove_shadows(img)
            metadata["remove_shadows"] = True

        if self.config.ENABLE_SCANNER_EFFECT and doc_type == "camera":
            img = auto_scan_document(img)

        if self.config.ENABLE_DESKEW:
            img, angle = deskew_image(img)
            if abs(angle) > 0.5:
                metadata["deskew_angle"] = angle
        if is_too_blurry(img):
            metadata["quality_warning"] = "blurry"

        if is_blank(img):
            metadata["quality_warning"] = "blank_page"
        if self.config.ENABLE_CROP:
            img = crop(img)

        if self.config.ENABLE_DENOISE:
            img = denoise(img)

        if self.config.ENABLE_RED_STAMP:
            erase = (doc_type == "scan")
            img, mask = red_stamp(img, erase=erase)
            metadata["has_red_stamp"] = bool(np.any(mask))
            metadata["red_stamp_mode"] = "erase" if erase else "keep"

        if self.config.ENABLE_BINARIZATION:
            img = binarize(img)

        img = normalize_for_ocr(img, ocr_engine)
        metadata["ocr_engine"] = ocr_engine

        return {
            "page_index": page_index,
            "image": img,
            "dpi": self.config.DPI,
            "width": img.shape[1],
            "height": img.shape[0],
            "metadata": metadata,
            "processing_time_ms": int((time.time() - t0)*1000),
            "status": "success"
        }
