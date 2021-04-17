"""
pytoch jit-traced backend 
"""
# pylint: disable=unused-argument,missing-docstring
import torch  # currently supports pytorch1.0
import backend


class BackendPytorchJITTraced(backend.Backend):
    def __init__(self):
        super(BackendPytorchJITTraced, self).__init__()
        self.sess = None
        self.model = None
        self.device = "cpu"

    def version(self):
        return torch.__version__

    def name(self):
        return "pytorch-jit-traced"

    def image_format(self):
        return "NCHW"

    def load(self, model_path, inputs=None, outputs=None):
        self.model = torch.jit.load(model_path, map_location='cpu')
        self.model.eval()
        # dummy
        self.inputs = ["input"]
        self.outputs = ["output"]
        # prepare the backend
        self.model = self.model.to(self.device)
        return self
        
    def predict(self, feed):
        key=[key for key in feed.keys()][0]    
        feed[key] = torch.tensor(feed[key]).float().to(self.device)
        with torch.no_grad():
            output = self.model(feed[key])    
        return [output]
