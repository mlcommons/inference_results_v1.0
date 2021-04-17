"""
pytoch fp32 backend 
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
from utils.general import non_max_suppression


class BackendPytorchYOLOv3FP32(backend.Backend):
    def __init__(self):
        super(BackendPytorchYOLOv3FP32, self).__init__()
        self.sess = None
        self.model = None
        # https://github.com/ultralytics/yolov3
        self.conf = 0.001
        self.iou = 0.65

    def version(self):
        return torch.__version__

    def name(self):
        return "pytorch-yolov3-fp32traced"

    def image_format(self):
        return "NCHW"

    def load(self, model_path, inputs=None, outputs=None):
        self.model = torch.jit.load(model_path)
        self.model.eval()

        # dummy
        self.inputs = ["input"]
        self.outputs = ["output"]
        return self
        
    def predict(self, feed):
        key=[key for key in feed.keys()][0]    
        feed[key] = torch.tensor(feed[key]).float()
        size = feed[key].shape[2]
        with torch.no_grad():
            pred = self.model(feed[key])[0]
        pred = non_max_suppression(pred, conf_thres=self.conf, iou_thres=self.iou)[0]
        bboxes = pred[..., :4]/size
        scores = pred[..., 4]
        labels = pred[..., 5].int()+1
        return [bboxes], [labels], [scores]

