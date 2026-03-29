import cv2
from PIL import Image
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

class TextRecognizer:
    def __init__(self, use_gpu: bool = False):
        config = Cfg.load_config_from_name('vgg_transformer')
        config['device'] = 'cuda' if use_gpu else 'cpu'
        config['predictor']['beamsearch'] = False
        self.model = Predictor(config)

    def recognize(self, cropped_img):
        pil_img = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
        return self.model.predict(pil_img, return_prob=True)