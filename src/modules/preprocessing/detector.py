import sys
import os
import cv2
import numpy as np

def detect_document_type(image: np.ndarray) -> str:
   
    h, w = image.shape[:2]

    small = cv2.resize(image, (400, int(400 * h / w)))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if len(small.shape) == 3 else small
        
    bg = cv2.dilate(gray, np.ones((15, 15), np.uint8))
    
    std_dev = np.std(bg)
    
    if std_dev < 15:
        return "scan"
    
    return "camera"
