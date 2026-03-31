import os
import sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
os.environ['USERPROFILE'] = PROJECT_ROOT
os.environ['HOME'] = PROJECT_ROOT

from paddleocr import PaddleOCR

class TextDetector:
    def __init__(self, use_gpu: bool = False):
        use_mkldnn = not use_gpu 
        self.model = PaddleOCR(
            use_angle_cls=True, 
            lang="vi", 
            use_gpu=use_gpu,
            use_mkldnn=use_mkldnn,
            cpu_threads=4,
            show_log=False,
            # 🚀 GIỚI HẠN VÙNG QUÉT: Tăng tốc cực độ cho PaddleOCR
            det_limit_side_len=960,
            det_limit_type='max'
        )

    def detect(self, img: np.ndarray):
        result = self.model.ocr(img, rec=False)
        return result[0] if result and result[0] else []

    def get_orientation(self, img: np.ndarray):
        result = self.model.ocr(img, det=False, rec=False, cls=True)
        if result and result[0]:
            return int(result[0][0][0]), float(result[0][0][1])
        return 0, 0.0