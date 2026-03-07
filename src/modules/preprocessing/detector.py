import cv2
import numpy as np

def detect_document_type(image: np.ndarray) -> str:
    """
    Phân loại ảnh đầu vào: scan | camera
    Không dùng AI.
    """
    h, w = image.shape[:2]

    # 1. Aspect ratio check (A4 ~ 1.29)
    ratio = max(h, w) / min(h, w)
    is_a4_like = 1.2 < ratio < 1.4

    # 2. Shadow variance
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    brightness_std = np.std(gray)

    # 3. Edge density
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = np.sum(edges > 0) / (h * w)

    # Quyết định
    if is_a4_like and brightness_std < 50 and edge_ratio < 0.05:
        return "scan"

    return "camera"
