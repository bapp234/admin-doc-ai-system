import cv2
import numpy as np

def crop_and_rotate_box(image, points):
    points = np.array(points, dtype=np.float32)
    
    width_top = np.linalg.norm(points[0] - points[1])
    width_bot = np.linalg.norm(points[2] - points[3])
    max_width = int(max(width_top, width_bot))
    
    height_left = np.linalg.norm(points[0] - points[3])
    height_right = np.linalg.norm(points[1] - points[2])
    max_height = int(max(height_left, height_right))
    
    dst_pts = np.array([
        [0, 0], [max_width - 1, 0], 
        [max_width - 1, max_height - 1], [0, max_height - 1]
    ], dtype="float32")
    
    M = cv2.getPerspectiveTransform(points, dst_pts)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))
    
    # 🔴 BÍ KÍP HÌNH HỌC CHỐNG LẬT ẢNH:
    # Nếu khung chữ cao hơn rộng (như sp2.jpg), tự động xoay ngang ngược chiều kim đồng hồ 90 độ
    if max_height > max_width * 1.2:
        warped = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    # Thêm viền trắng giúp VietOCR đọc chuẩn hơn
    padding = 2
    warped = cv2.copyMakeBorder(
        warped, padding, padding, padding, padding, 
        cv2.BORDER_CONSTANT, value=[255, 255, 255]
    )
    
    return warped