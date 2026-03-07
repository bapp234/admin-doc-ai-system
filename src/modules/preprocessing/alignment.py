import cv2
import numpy as np
from loguru import logger

def order_points(pts):
    """Sắp xếp 4 điểm góc: Top-Left, Top-Right, Bottom-Right, Bottom-Left"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    """Biến đổi phối cảnh"""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # Tính kích thước mới
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

def auto_scan_document(image: np.ndarray) -> np.ndarray:
    """
    Thuật toán tìm khung giấy bằng Canny Edge Detection.
    Không dùng AI.
    """
    try:
        # Tính diện tích ảnh gốc
        h_orig, w_orig = image.shape[:2]
        area_orig = h_orig * w_orig
        
        ratio = h_orig / 500.0
        orig = image.copy()
        image_small = cv2.resize(image, (int(w_orig / ratio), 500))

        gray = cv2.cvtColor(image_small, cv2.COLOR_BGR2GRAY)
        
        # [TWEAK] Dùng Bilateral để giữ cạnh nhưng xóa nhiễu tốt hơn Gaussian
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        edged = cv2.Canny(gray, 30, 200) # Threshold rộng hơn

        cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts[0] if len(cnts) == 2 else cnts[1], key=cv2.contourArea, reverse=True)[:5]

        screenCnt = None
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            if len(approx) == 4:
                # [FIX QUAN TRỌNG] Kiểm tra diện tích
                # Nếu vùng tìm được < 20% ảnh gốc -> Bỏ qua (đó là rác hoặc bảng biểu)
                # Tính diện tích contour trên ảnh gốc (nhân lại ratio bình phương)
                contour_area = cv2.contourArea(approx) * (ratio * ratio)
                
                if contour_area > (0.2 * area_orig): # Ngưỡng 20%
                    screenCnt = approx
                    break

        if screenCnt is not None:
            logger.info("✅ Đã tìm thấy khung giấy hợp lệ.")
            warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
            return warped
        
        logger.debug("⚠️ Không tìm thấy khung giấy đủ lớn. Giữ nguyên ảnh gốc.")
        return image

    except Exception as e:
        logger.warning(f"Lỗi Alignment: {e}")
        return image