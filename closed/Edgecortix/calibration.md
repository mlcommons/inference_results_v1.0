# MLPerf Inference v1.0 - Calibration

Edgecortix quantize and calibrate FP32 precision models using PyTorch's built-in post-training static quantization framework.
Weights are quantized per-channel to 8 bit precision int8_t and activations to 8 bit precision uint8_t using the FBGemm quantization back-end.

Calibration script [calibrate_torchvision_model.py](calibrate_torchvision_model.py).

For more information about Pytorch post-trainning static quantization please refer to this [document](https://pytorch.org/blog/introduction-to-quantization-on-pytorch/#post-training-static-quantization) for a more detailed explanation.
