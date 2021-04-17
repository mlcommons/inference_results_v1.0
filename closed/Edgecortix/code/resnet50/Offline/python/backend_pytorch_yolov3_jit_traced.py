"""
pytoch jit-traced backend 
"""
# pylint: disable=unused-argument,missing-docstring
import json
import os
import time
import torch 
import torch.nn as nn
import torchvision
import backend

import sys
sys.path.append('yolov3')
from models.yolo import Detect
from utils.general import non_max_suppression


config = {
    'yolov3-jit': (
        [
            [10, 13, 16, 30, 33, 23],
            [30, 61, 62, 45, 59, 119],
            [116, 90, 156, 198, 373, 326]
        ],
        [256,512, 1024],
    ),
    'yolov3-tiny-jit': (
        [
            [10, 14, 23, 27, 37, 58],
            [81, 82, 135, 169, 344, 319]
        ],
        [256, 512],
    ),
}


class BackendPytorchYOLOv3JITTraced(backend.Backend):
    def __init__(self):
        super(BackendPytorchYOLOv3JITTraced, self).__init__()
        self.sess = None
        self.model = None
        # https://github.com/ultralytics/yolov3
        self.conf = 0.001
        self.iou = 0.65

    def version(self):
        return torch.__version__

    def name(self):
        return "pytorch-yolov3-jit-traced"

    def image_format(self):
        return "NCHW"

    def load(self, model_path, inputs=None, outputs=None):
        self.model = torch.jit.load(model_path)
        model_name = os.path.split(model_path)[-1].replace('.pt', '')
        anchors, ch = config[model_name]
        pp = Detect(80, anchors, ch)
        s = 128
        pp.stride = torch.tensor([s / x.shape[-2] for x in self.model(torch.zeros(1, 3, s, s))])
        pp.anchors /= pp.stride.view(-1, 1, 1)
        pp.load_state_dict(torch.load(model_path.replace('.pt', '-pp.pt')))
        pp.eval()
        self.post_processor = pp

        # dummy
        self.inputs = ["input"]
        self.outputs = ["output"]
        return self
        
    def predict(self, feed):
        key=[key for key in feed.keys()][0]    
        feed[key] = torch.tensor(feed[key]).float()
        size = feed[key].shape[2]
        with torch.no_grad():
            output = self.model(feed[key])
            pred = self.post_processor(list(output))[0]
        pred = non_max_suppression(pred, conf_thres=self.conf, iou_thres=self.iou)[0]
        bboxes = pred[..., :4]/size
        scores = pred[..., 4]
        labels = pred[..., 5].int()+1
        return [bboxes], [labels], [scores]

