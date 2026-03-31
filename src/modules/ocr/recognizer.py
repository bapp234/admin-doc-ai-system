import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
import cv2
import torch
from PIL import Image
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from loguru import logger

class TextRecognizer:
    def __init__(self, use_gpu: bool = False):
        # 🚀 TRỞ VỀ VGG_TRANSFORMER ĐỂ ĐẢM BẢO KHÔNG BỊ SAI CHÍNH TẢ
        config = Cfg.load_config_from_name('vgg_transformer')
        config['device'] = 'cuda' if use_gpu else 'cpu'
        
        # Bật lại Beamsearch (Sửa lỗi "đoán mò" thiếu chính xác của AI)
        config['predictor']['beamsearch'] = True
        
        self.model = Predictor(config)
        self.batch_size = 32 if use_gpu else 8 

    def recognize_batch(self, cropped_imgs: list):
        if not cropped_imgs: return [], []
            
        pil_imgs = []
        for img in cropped_imgs:
            if img is not None and img.size > 0:
                pil_imgs.append(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))

        all_texts, all_probs = [], []
        
        with torch.no_grad():
            if hasattr(self.model, 'predict_batch'):
                for i in range(0, len(pil_imgs), self.batch_size):
                    batch = pil_imgs[i : i + self.batch_size]
                    try:
                        t, p = self.model.predict_batch(batch, return_prob=True)
                        all_texts.extend(t)
                        all_probs.extend(p)
                    except Exception as e:
                        for img in batch:
                            t_seq, p_seq = self.model.predict(img, return_prob=True)
                            all_texts.append(t_seq)
                            all_probs.append(p_seq)
            else:
                for pil_img in pil_imgs:
                    t, p = self.model.predict(pil_img, return_prob=True)
                    all_texts.append(t)
                    all_probs.append(p)
                
        return all_texts, all_probs