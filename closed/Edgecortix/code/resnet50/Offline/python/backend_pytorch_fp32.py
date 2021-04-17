"""
pytoch FP32 backend 
"""
# pylint: disable=unused-argument,missing-docstring
import torch  # currently supports pytorch1.0
import backend


class BackendPytorchFP32(backend.Backend):
    def __init__(self):
        super(BackendPytorchFP32, self).__init__()
        self.sess = None
        self.model = None
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def version(self):
        return torch.__version__

    def name(self):
        return "pytorch-fp32"

    def image_format(self):
        return "NCHW"

    def load(self, model_path, inputs=None, outputs=None):
        self.model = torch.load(model_path, map_location='cpu')
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
        return [output.cpu()]
