import cv2
import numpy as np

def crop(image: np.ndarray) -> np.ndarray:
    """Cắt viền đen thừa"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(thresh)
    
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        pad = 10
        return image[max(0, y-pad):min(image.shape[0], y+h+pad), 
                     max(0, x-pad):min(image.shape[1], x+w+pad)]
    return image