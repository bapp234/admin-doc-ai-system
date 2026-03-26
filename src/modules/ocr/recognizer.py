from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from PIL import Image
import cv2

class TextRecognizer:
    def __init__(self, use_gpu=False):
        # Cấu hình vgg_transformer là bản ổn định nhất cho tiếng Việt
        config = Cfg.load_config_from_name('vgg_transformer')
        config['device'] = 'cuda' if use_gpu else 'cpu'
        config['predictor']['beamsearch'] = False
        self.model = Predictor(config)

    def recognize(self, cropped_img):
        """Nhận diện chữ từ mảnh ảnh đã cắt."""
        # VietOCR cần định dạng PIL Image
        pil_img = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
        text, prob = self.model.predict(pil_img, return_prob=True)
        return text, prob