import sys
import os
import cv2
import numpy as np

def denoise(image: np.ndarray) -> np.ndarray:
    
    return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)

def binarize(image: np.ndarray) -> np.ndarray:
   
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    return cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
def remove_shadows(image: np.ndarray) -> np.ndarray:
   
    h, w=image.shape[:2]

    small_img = cv2.resize(image, (w // 4, h // 4))
    planes = cv2.split(image)
    small_planes = cv2.split(small_img)
    result_planes = []
    
    for i in range(len(planes)):        
        dilated_small = cv2.dilate(small_planes[i], np.ones((7, 7), np.uint8))
        bg_small = cv2.medianBlur(dilated_small, 21)
        
        bg_large = cv2.resize(bg_small, (w, h))
        
        diff_img = 255 - cv2.absdiff(planes[i], bg_large)
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        if np.mean(norm_img) > 230:
             gamma = 0.5
             invGamma = 1.0 / gamma
             table = np.array([((j / 255.0) ** invGamma) * 255 for j in np.arange(0, 256)]).astype("uint8")
             norm_img = cv2.LUT(norm_img, table)

        result_planes.append(norm_img)
        
    return cv2.merge(result_planes)

def denoise(image: np.ndarray) -> np.ndarray:
    return cv2.bilateralFilter(image, 9, 75, 75)

def binarize(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 10)