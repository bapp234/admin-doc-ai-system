import numpy as np
from paddleocr import PaddleOCR

class TextDetector:
    def __init__(self, use_gpu: bool = False):
        self.model = PaddleOCR(use_angle_cls=True, lang="vi", use_gpu=use_gpu, show_log=False)

    def detect(self, img: np.ndarray):
        result = self.model.ocr(img, rec=False)
        return result[0] if result and result[0] else []

    def get_orientation(self, img: np.ndarray):
        result = self.model.ocr(img, det=False, rec=False, cls=True)
        if result and result[0]:
            return int(result[0][0][0]), float(result[0][0][1])
        return 0, 0.0