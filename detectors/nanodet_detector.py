import cv2
import torch
import numpy as np
from detectors.BaseDetector import baseDet


from models.nanodet.data.batch_process import stack_batch_img
from models.nanodet.data.collate import naive_collate
from models.nanodet.data.transform import Pipeline
from models.nanodet.model.arch import build_model
from models.nanodet.util import Logger, cfg, load_config, load_model_weight
from models.nanodet.util.path import mkdir

class Detector(baseDet):

    def __init__(self):
        super(Detector, self).__init__()
        self.init_model()
        self.build_config()

    def init_model(self):

        self.weights = 'weights/nanodet_model_best.pth'
        self.config = 'models/nanodet/nanodet_custom_xml_dataset_person.yml'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger = Logger(0, use_tensorboard=False)

        load_config(cfg, self.config)
        self.cfg = cfg

        model = build_model(cfg.model)
        ckpt = torch.load(self.weights, map_location=lambda storage, loc: storage)
        load_model_weight(model, ckpt, logger)
        if self.cfg.model.arch.backbone.name == "RepVGG":
            deploy_config = self.cfg.model
            deploy_config.arch.backbone.update({"deploy": True})
            deploy_model = build_model(deploy_config)
            from models.nanodet.model.backbone.repvgg import repvgg_det_model_convert

            model = repvgg_det_model_convert(model, deploy_model)
        self.model = model.to(self.device).eval()
        self.pipeline = Pipeline(self.cfg.data.val.pipeline, self.cfg.data.val.keep_ratio)
        self.names = ['Person']

    def preprocess(self, img):

        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        meta = dict(img_info=img_info, raw_img=img, img=img)
        meta = self.pipeline(None, meta, self.cfg.data.val.input_size)
        meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1)).to(self.device)
        meta = naive_collate([meta])
        meta["img"] = stack_batch_img(meta["img"], divisible=32)

        return meta

    def detect(self, im):

        meta = self.preprocess(im)

        with torch.no_grad():
            results = self.model.inference(meta)

        pred = results[0]

        pred_boxes = []
        for label in pred:
            for bbox in pred[label]:
                score = bbox[-1]
                if score > 0.45:
                    x0, y0, x1, y1 = [int(i) for i in bbox[:4]]
                    pred_boxes.append([x0, y0, x1, y1, self.names[label], score])
        pred_boxes.sort(key=lambda v: v[5])
        return im, pred_boxes

