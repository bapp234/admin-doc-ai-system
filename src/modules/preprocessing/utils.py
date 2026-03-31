import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
import cv2
def ensure_gray(img):
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def ensure_bgr(img):
    if len(img.shape) == 3:
        return img
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
