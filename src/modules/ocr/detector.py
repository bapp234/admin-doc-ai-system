from paddleocr import PaddleOCR
import numpy as np

class TextDetector:
    def __init__(self, use_gpu=False):
        # Khởi tạo PaddleOCR chuyên cho nhiệm vụ Detection và Classification (Hướng)
        self.model = PaddleOCR(
            use_angle_cls=True, 
            lang="vi", 
            use_gpu=use_gpu, 
            show_log=False
        )

    def detect(self, img):
        """Trả về danh sách các boxes tọa độ."""
        result = self.model.ocr(img, rec=False)
        return result[0] if result and result[0] else []

    def get_orientation(self, img):
        """Xác định góc xoay của ảnh (0, 90, 180, 270)."""
        result = self.model.ocr(img, det=False, rec=False, cls=True)
        if result and result[0]:
            angle_label = result[0][0][0]
            score = result[0][0][1]
            return int(angle_label), score
        return 0, 0.0