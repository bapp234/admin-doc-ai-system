import cv2
import numpy as np
from typing import Tuple
from loguru import logger

def deskew_image(
    img: np.ndarray,
    max_angle: float = 10.0,
    angle_step: float = 0.5
) -> Tuple[np.ndarray, float]:
    if img is None or img.size == 0: return img, 0.0
    scale = 1.0
    h, w = img.shape[:2]
    if h < 200 or w < 200: return img, 0.0

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    working_img = gray
    if w > 1000:
        scale = 1000 / w
        new_h = int(h * scale)
        working_img = cv2.resize(gray, (1000, new_h), interpolation=cv2.INTER_AREA)

    blurred = cv2.GaussianBlur(working_img, (5, 5), 0)
    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    best_angle = 0.0
    best_score = -1.0
    wh, ww = thresh.shape
    center = (ww // 2, wh // 2)

    for angle in np.arange(-max_angle, max_angle + angle_step, angle_step):
        if abs(angle) < 1e-6:
            rotated = thresh
        else:
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(
                thresh, M, (ww, wh), 
                flags=cv2.INTER_NEAREST, 
                borderMode=cv2.BORDER_CONSTANT, 
                borderValue=0
            )
        
        score = np.var(np.sum(rotated, axis=1))
        if score > best_score:
            best_score = score
            best_angle = angle

    if abs(best_angle) > 0.2:
        logger.info(f"Góc nghiêng: {best_angle:.2f} độ")
        M = cv2.getRotationMatrix2D((w // 2, h // 2), best_angle, 1.0)
        border_bg = (255, 255, 255) if len(img.shape) == 3 else 255
        deskewed = cv2.warpAffine(
            img, M, (w, h), 
            flags=cv2.INTER_CUBIC, 
            borderMode=cv2.BORDER_CONSTANT, 
            borderValue=border_bg
        )
        return deskewed, best_angle
    
    return img, 0.0