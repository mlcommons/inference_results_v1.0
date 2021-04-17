"""
pytoch Edgecortix backend 
"""
# pylint: disable=unused-argument,missing-docstring
import os
import torch
import backend
import ip_runtime
import torchvision
import torchvision.transforms as transforms
import tvm

from tvm import relay
from tvm.relay import mera
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader, Dataset

class CalibrationDataset(Dataset):
    def __init__(self, root, files, transform):
        with open(files, 'r') as f:
            self.files = [os.path.join(root, fn.strip()) for fn in f.readlines()]
        self.transform = transform

    def __getitem__(self, idx):
        image = Image.open(self.files[idx]).convert('RGB')
        image = self.transform(image)
        return image

    def __len__(self):
        return len(self.files)


class BackendEdgecortix(backend.Backend):
    def __init__(self, dataset_path, dataset_calibration_list):
        super(BackendEdgecortix, self).__init__()
        self.sess = None
        self.model = None
        self.iprt = None
        self.device = "cpu"
        self.dataset_path = dataset_path
        self.dataset_calibration_list = dataset_calibration_list

    def version(self):
        return ""

    def name(self):
        return "edgecortix"

    def image_format(self):
        return "NHWC"

    def quantize_model(self, transform, quantization_backend='fbgemm'):
        print(torch.backends.quantized.supported_engines)
        print(quantization_backend)
        if quantization_backend not in torch.backends.quantized.supported_engines:
            raise RuntimeError("Quantized backend not supported ")
        torch.backends.quantized.engine = quantization_backend
        self.model.cpu()
        self.model.eval()
        self.model.fuse_model()

        dataset = CalibrationDataset(root=self.dataset_path, files=self.dataset_calibration_list, transform=transform)
        dataloader = DataLoader(dataset, batch_size=1) 

        self.model.qconfig = torch.quantization.get_default_qconfig(quantization_backend)
        torch.quantization.prepare(self.model, inplace=True)
        for x in tqdm(dataloader):
            self.model(x)
            break
        torch.quantization.convert(self.model, inplace=True)

    def compile_model(self, input_shape, torch_input_shape, output_dir, config):
        inputs = [("input0", input_shape)]
        input_layout = self.image_format()
        with torch.no_grad():
            traced_model = torch.jit.trace(self.model, torch.rand(torch_input_shape)).eval()
        mod, params = relay.frontend.from_pytorch(traced_model, inputs, layout=input_layout)
        with mera.build_config(target="IP", **config):
            mera.build(mod, params, output_dir=output_dir, host_arch="x86", layout=input_layout)

    def load(self, model_path, inputs=None, outputs=None):
        arch = {"arch": 400}
        if model_path == 'torchvision-resnet50':
            self.model = torchvision.models.quantization.resnet50(pretrained=True, progress=True, quantize=False)
            self.model.eval()
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            input_shape = (1, 224, 224, 3)  # NHWC
            torch_input_shape = (1, 3, 224, 224)  # NCHW
        elif model_path == 'torchvision-mobilenetv2':
            self.model = torchvision.models.quantization.mobilenet.mobilenet_v2(pretrained=True, progress=True, quantize=False)
            self.model.eval()
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            input_shape = (1, 224, 224, 3)  # NHWC
            torch_input_shape = (1, 3, 224, 224)  # NCHW
            arch = {
                "arch": 401,
                "scheduler_config": {"mode": "Fast"}
            }
        else:
            raise RuntimeError("Preset model not available: ", model_path)

        ec_dir = "./ec-" + model_path
        if not os.path.exists(ec_dir):
            self.quantize_model(transform)
            self.compile_model(input_shape, torch_input_shape, ec_dir, arch)

        self.iprt = ip_runtime.IPRuntime()
        self.iprt.Setup(ec_dir)
        # dummy
        self.inputs = ["input"]
        self.outputs = ["output"]
        return self

    def predict(self, feed):
        key=[key for key in feed.keys()][0]    
        output_ip = torch.from_numpy(self.iprt.Run(feed[key])[0])
        return [output_ip]

