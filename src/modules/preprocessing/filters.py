import cv2
import numpy as np

def denoise(image: np.ndarray) -> np.ndarray:
    """Khử nhiễu"""
    # fastNlMeansDenoising chạy hơi chậm nhưng kết quả đẹp
    # Nếu muốn nhanh hơn dùng: return cv2.GaussianBlur(image, (3, 3), 0)
    return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)

def binarize(image: np.ndarray) -> np.ndarray:
    """Adaptive Thresholding"""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    return cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
def remove_shadows(image: np.ndarray) -> np.ndarray:
    """
    Kỹ thuật xóa bóng chuyên sâu cho tài liệu.
    """
    # 1. Tách kênh màu
    planes = cv2.split(image)
    result_planes = []
    
    for plane in planes:
        # 2. Tạo ảnh nền (Background) bằng Dilate + MedianBlur
        # Kernel càng to thì xóa bóng càng mạnh, nhưng cẩn thận mất chữ to
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        
        # 3. Phép chia chuẩn hóa (Quan trọng nhất)
        # Công thức: Result = 255 * (Image / Background)
        # Chuyển sang float để tính toán chính xác
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        
        # Chuẩn hóa về 0-255
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # [FIX SP3 TRẮNG XÓA]
        # Nếu ảnh sau khi xử lý quá nhạt (trắng hếu), ta dùng Gamma Correction để kéo đậm lại
        # Kiểm tra độ sáng trung bình
        mean_brightness = np.mean(norm_img)
        if mean_brightness > 230: # Ảnh quá trắng
             # Gamma < 1.0 làm ảnh tối đi (đậm chữ hơn)
             gamma = 0.5
             invGamma = 1.0 / gamma
             table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
             norm_img = cv2.LUT(norm_img, table)

        result_planes.append(norm_img)
        
    return cv2.merge(result_planes)

def denoise(image: np.ndarray) -> np.ndarray:
    # Bước xóa bóng tích hợp luôn vào đây
    image = remove_shadows(image)
    # Khử nhiễu nhẹ
    return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)

def binarize(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Dùng Gaussian Adaptive Threshold
    # blockSize=15, C=10 (C càng lớn thì càng lọc mạnh nhiễu, nền càng trắng)
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 10)