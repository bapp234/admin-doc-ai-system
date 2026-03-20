import cv2
import numpy as np

def detect_document_type(image: np.ndarray) -> str:
    """
    Phân loại ảnh đầu vào: scan | camera dựa trên độ lệch chuẩn của ánh sáng nền.
    """
    h, w = image.shape[:2]

    # 1. Resize thật nhỏ (chiều ngang 400px) để tính trong 10ms
    small = cv2.resize(image, (400, int(400 * h / w)))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if len(small.shape) == 3 else small
        
    # 2. Dùng phép nong (Dilate) bôi đen hết chữ, chỉ giữ lại màu của tờ giấy
    bg = cv2.dilate(gray, np.ones((15, 15), np.uint8))
    
    # 3. Tính độ lệch chuẩn (Mức độ gồ ghề của ánh sáng)
    std_dev = np.std(bg)
    
    # Scan xịn có nền phẳng lỳ (độ lệch chuẩn rất thấp)
    if std_dev < 15:
        return "scan"
    
    return "camera"
