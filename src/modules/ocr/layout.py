import numpy as np
from doctr.models import ocr_predictor

class LayoutAnalyzer:
    def __init__(self, use_gpu=True):
        self.predictor = ocr_predictor(
            det_arch='db_resnet50', 
            reco_arch='crnn_vgg16_bn', 
            pretrained=True
        )

    def get_reading_order(self, img_array):
        # docTR yêu cầu ảnh hệ màu RGB
        rgb_img = img_array[:, :, ::-1]
        doc = self.predictor([rgb_img])
        
        h, w = img_array.shape[:2]
        ordered_boxes = []
        
        # Bóc tách cấu trúc: Page -> Block -> Line (Đảm bảo thứ tự đọc chuẩn xác)
        for page in doc.pages:
            for block in page.blocks:
                for line in block.lines:
                    geom = line.geometry 
                    xmin, ymin = geom[0]
                    xmax, ymax = geom[1]
                    
                    x1, y1 = int(xmin * w), int(ymin * h)
                    x2, y2 = int(xmax * w), int(ymax * h)
                    
                    box = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                    ordered_boxes.append(box)
                    
        return ordered_boxes