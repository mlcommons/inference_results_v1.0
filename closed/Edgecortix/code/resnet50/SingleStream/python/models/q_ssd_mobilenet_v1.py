import torch
import torch.nn as nn
from collections import OrderedDict
from torch.quantization import QuantStub, DeQuantStub, fuse_modules
from models.ssd_mobilenet_v1 import Block, MobileNetV1Base, PredictionHead
from models.anchor_generator import create_ssd_anchors
from models.utils import BiasAdd, BatchNorm2d, Conv2d_tf, nms, decode_boxes


class ZeroPad2d_Contiguous(nn.ZeroPad2d):
    def __init__(self, padding):
        super().__init__(padding)

    def forward(self, x):
        x = x.contiguous()
        x = super().forward(x)
        return x


def convert_Conv2d_tf(m, inp):
    zero_pad = None
    if m.padding=='VALID':
        padding = 0
    else:
        rows_odd, padding_rows = m._compute_padding(inp, dim=0)
        cols_odd, padding_cols = m._compute_padding(inp, dim=1)
        if rows_odd or cols_odd:
            zero_pad = ZeroPad2d_Contiguous([0, cols_odd, 0, rows_odd])
        padding = (padding_cols//2, padding_rows//2)
    bias = False if m.bias is None else True
    conv = nn.Conv2d(
        m.in_channels,
        m.out_channels,
        kernel_size=m.kernel_size,
        stride=m.stride,
        padding=padding,
        dilation=m.dilation,
        bias=bias,
        groups=m.groups)
    # copy weights and biases into Conv2d
    conv.weight = nn.Parameter(m.weight.data)
    if bias:
        conv.bias = nn.Parameter(m.bias.data)
    return conv, zero_pad


def convert_BiasAdd(m):
    # copy bias into BatchNorm2d
    bn = nn.BatchNorm2d(m.bias.shape[0], momentum=1.0, eps=0)
    bn.bias = nn.Parameter(m.bias.data)
    return bn


def convert_BatchNorm2d(m):
    # copy weight and bias into BatchNorm2d
    bn = nn.BatchNorm2d(m.bias.shape[0], momentum=1.0, eps=0)
    bn.weight = nn.Parameter(m.scale.data)
    bn.bias = nn.Parameter(m.bias.data)
    return bn


def convert_non_standard_modules(model):
    # convert backbone modules
    backbone_modules = []
    def backbone_hook(module, inp, out):
        ops = []
        inp = inp[0]
        for n, m in module.named_children():
            if isinstance(m, Conv2d_tf):
                m, zp = convert_Conv2d_tf(m, inp)
                if zp is not None:
                    ops.append((n + '/zero_pad', zp))
                    inp = zp(inp)
            elif isinstance(m, BiasAdd):
                m = convert_BiasAdd(m)
            elif isinstance(m, BatchNorm2d):
                m = convert_BatchNorm2d(m)
            # converting ReLU6 to ReLU to fuse drops mAP
            #elif isinstance(m, nn.ReLU6):
            #    m = nn.ReLU()
            m.eval()
            ops.append((n, m))
            inp = m(inp)
        backbone_modules.append(nn.Sequential(OrderedDict(ops)))

    # convert extras modules
    extras_modules = []
    def extras_hook(module, inp, out):
        ops = []
        inp = inp[0]
        for m in module.children():
            if isinstance(m, Conv2d_tf):
                m, zp = convert_Conv2d_tf(m, inp)
                if zp is not None:
                    ops.append(zp)
                    inp = zp(inp)
            # converting ReLU6 to ReLU to fuse drops mAP
            #elif isinstance(m, nn.ReLU6):
            #    m = nn.ReLU()
            m.eval()
            ops.append(m)
            inp = m(inp)
        extras_modules.append(nn.Sequential(*ops))

    hooks = []
    for m in model.backbone.children():
        hooks.append(m.register_forward_hook(backbone_hook))
    for m in model.extras.children():
        hooks.append(m.register_forward_hook(extras_hook))

    model.eval()
    with torch.no_grad():
        inp = torch.rand(1, 3, 300, 300)
        model(inp)

    for hook in hooks:
        hook.remove()

    # update backbone and extras
    for i, module in enumerate(backbone_modules):
        model.backbone[i] = module
    for i, module in enumerate(extras_modules):
        model.extras[i] = module


class QuantizableMobileNetV1Base(MobileNetV1Base):
    def __init__(self, return_layers=[11, 13]):
        super().__init__(return_layers)

    def fuse_model(self):
        for m in self.children():
            names = []
            for n, _ in m.named_children():
                names.append(n)
            for i in range(len(m)-1):
                if isinstance(m[i], nn.Conv2d) and isinstance(m[i+1], nn.BatchNorm2d):
                    if i+2 < len(m) and isinstance(m[i+2], nn.ReLU):
                        fuse_modules(m, [names[i], names[i+1], names[i+2]], inplace=True)
                    else:
                        fuse_modules(m, [names[i], names[i+1]], inplace=True)


class QuantizableExtras(nn.ModuleList):
    def __init__(self):
        super().__init__([
            Block(1024, 256, 512),
            Block(512, 128, 256),
            Block(256, 128, 256),
            Block(256, 64, 128),
        ])

    def fuse_model(self):
        for m in self.children():
            for i in range(len(m)-1):
                if isinstance(m[i], nn.Conv2d) and isinstance(m[i+1], nn.ReLU):
                    fuse_modules(m, [str(i), str(i+1)], inplace=True)

    def forward(self, x):
        out = []
        for module in self.children():
            x = module(x)
            out.append(x)
        return out


class Predictors(nn.ModuleList):
    def __init__(self, num_classes):
        super().__init__([
            PredictionHead(in_channels, num_classes, num_anchors)
            for in_channels, num_anchors in zip(
                (512, 1024, 512, 256, 256, 128), (3, 6, 6, 6, 6, 6)
            )
        ])


class QuantizableSSD(nn.Module):
    def __init__(self, backbone, predictors, extras):
        super().__init__()
        self.backbone = backbone
        self.predictors = predictors
        self.extras = extras
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def fuse_model(self):
        convert_non_standard_modules(self)
        self.eval()
        self.backbone.fuse_model()
        self.extras.fuse_model()

    def forward(self, x):
        x = self.quant(x)
        feature_maps = self.backbone(x)

        out = feature_maps[-1]
        feature_maps.extend(self.extras(out))

        results = []
        for feature, module in zip(feature_maps, self.predictors):
            class_logits, box_regression = module(feature)
            results.append((self.dequant(class_logits), self.dequant(box_regression)))

        class_logits, box_regression = list(zip(*results))
        class_logits = torch.cat(class_logits, 1)
        box_regression = torch.cat(box_regression, 1)

        scores = torch.sigmoid(class_logits)
        box_regression = box_regression.squeeze(0)

        return scores, box_regression


class PostProcessor(nn.Module):
    def __init__(self, shapes=[(19, 19), (10, 10), (5, 5), (3, 3), (2, 2), (1, 1)]):
        super().__init__()

        self.priors = torch.cat(create_ssd_anchors()._generate(shapes), dim=0)

        # preprocess
        self.image_size = 300
        self.image_mean = 127.5
        self.image_std = 127.5

        self.coder_weights = torch.tensor((10, 10, 5, 5), dtype=torch.float32)

        # generate anchors for the sizes of the feature map

        # postprocess
        self.nms_threshold = 0.6

        # set it to 0.01 for better results but slower runtime
        self.score_threshold = 0.3

    def forward(self, scores, box_regression):
        if box_regression.dim()==2:
            box_regression = box_regression[None]
        boxes = decode_boxes(box_regression, self.priors, self.coder_weights)
        list_boxes=[]; list_labels=[]; list_scores=[]
        for b in range(len(scores)):
            bboxes, blabels, bscores = self.filter_results(scores[b], boxes[b])
            list_boxes.append(bboxes)
            list_labels.append(blabels.long())
            list_scores.append(bscores)
        #boxes = self.rescale_boxes(boxes, height, width)
        return [list_boxes, list_labels, list_scores]

    def filter_results(self, scores, boxes):
        # in order to avoid custom C++ extensions
        # we use an NMS implementation written purely
        # on python. This implementation is faster on the
        # CPU, which is why we run this part on the CPU
        cpu_device = torch.device("cpu")
        #boxes = boxes[0]
        #scores = scores[0]
        boxes = boxes.to(cpu_device)
        scores = scores.to(cpu_device)
        selected_box_probs = []
        labels = []
        for class_index in range(1, scores.size(1)):
            probs = scores[:, class_index]
            mask = probs > self.score_threshold
            probs = probs[mask]
            subset_boxes = boxes[mask, :]
            box_probs = torch.cat([subset_boxes, probs.reshape(-1, 1)], dim=1)
            box_probs = nms(box_probs, self.nms_threshold)
            selected_box_probs.append(box_probs)
            labels.append(
                torch.full((box_probs.size(0),), class_index, dtype=torch.int64)
            )
        selected_box_probs = torch.cat(selected_box_probs)
        labels = torch.cat(labels)
        return selected_box_probs[:, :4], labels, selected_box_probs[:, 4]

    def rescale_boxes(self, boxes, height, width):
        boxes[:, 0] *= width
        boxes[:, 1] *= height
        boxes[:, 2] *= width
        boxes[:, 3] *= height
        return boxes


def create_quantizable_mobilenetv1_ssd(num_classes):
    backbone = QuantizableMobileNetV1Base()
    extras = QuantizableExtras()
    predictors = Predictors(num_classes)
    return QuantizableSSD(backbone, predictors, extras)

