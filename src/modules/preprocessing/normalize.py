import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
import cv2
from .utils import ensure_gray, ensure_bgr

def normalize_for_ocr(img, engine: str):
    engine = engine.lower()

    if engine == "tesseract":
        gray = ensure_gray(img)
        return cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1]

    if engine == "paddle":
        return ensure_gray(img)

    if engine == "trocr":
        return ensure_bgr(img)

    return img
