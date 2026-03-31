import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
import numpy as np
import cv2

def is_too_blurry(img, threshold=80) -> bool:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold


def is_too_small(img, min_size=300) -> bool:
    h, w = img.shape[:2]
    return min(h, w) < min_size


def is_blank(img, ratio=0.98) -> bool:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    white_ratio = np.mean(gray > 245)
    return white_ratio > ratio
