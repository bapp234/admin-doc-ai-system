import sys
import os
import cv2
import numpy as np
from typing import Tuple

def red_stamp(image: np.ndarray, erase: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    if image.ndim < 3:
        return image, np.zeros(image.shape[:2], dtype=np.uint8)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower1 = np.array([0, 80, 80])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([170, 80, 80])
    upper2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    final_mask = np.zeros_like(mask)

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        aspect = w / float(h)

        if area > 800 and 0.6 < aspect < 1.4:
            final_mask[labels == i] = 255

    if not erase:
        return image, final_mask

    cleaned = image.copy()
    cleaned[final_mask == 255] = (255, 255, 255)

    return cleaned, final_mask
