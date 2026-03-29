import cv2
import numpy as np

def crop_and_rotate_box(image, points):
    points = np.array(points, dtype=np.float32)
    width = int(max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])))
    height = int(max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])))
    
    dst_pts = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(points, dst_pts)
    warped = cv2.warpPerspective(image, M, (width, height))
    
    if height > width * 1.2:
        warped = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    return cv2.copyMakeBorder(warped, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[255, 255, 255])