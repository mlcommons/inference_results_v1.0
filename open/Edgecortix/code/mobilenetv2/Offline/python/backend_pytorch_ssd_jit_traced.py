"""
pytoch jit-traced backend 
"""
# pylint: disable=unused-argument,missing-docstring
import torch  # currently supports pytorch1.0
import backend
from models.q_ssd_mobilenet_v1 import PostProcessor


class BackendPytorchSSDJITTraced(backend.Backend):
    def __init__(self):
        super(BackendPytorchSSDJITTraced, self).__init__()
        self.sess = None
        self.model = None
        self.post_processor = PostProcessor()

    def version(self):
        return torch.__version__

    def name(self):
        return "pytorch-ssd-jit-traced"

    def image_format(self):
        return "NCHW"

    def load(self, model_path, inputs=None, outputs=None):
        self.model = torch.jit.load(model_path, map_location='cpu')
        # dummy
        self.inputs = ["input"]
        self.outputs = ["output"]
        return self
        
    def predict(self, feed):
        key=[key for key in feed.keys()][0]    
        feed[key] = torch.tensor(feed[key]).float()
        with torch.no_grad():
            output = self.post_processor(*self.model(feed[key]))
        return output
