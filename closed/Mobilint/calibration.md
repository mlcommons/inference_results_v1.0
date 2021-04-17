# Mobilint FPGA Post-training Quantization 

Mobilint employs per-channel symmetric quantization for weight tensors and per-tensor symmetric quantization for activation tensors.  

## Activation 

A per-tensor symmetric quantization is used. Collecting activation statistic with calibration datasets listed in mlperf. Test 100%, 99.999%, 99.99% percentile values and choose best accuracy value for quantization(Mobilenet-SSD: 100%, Resnet34-SSD: 99.999%, Resnet50: 99.99%). Activation tensors were quantized to int8. 

## Weights 

A per-channel symmetric quantization is used. First, find maximum absolute value of each output channel of the weight tensors. Second, calculate mse(mean square error) between fp32 weight tensors and quantized weight tensors with several quantization scale(maximum absolute value * [0.95:1.05:0.01] / 127). Finally, choose minimum mse value and quantization. Weight tensors were quantized to int8 and bias tensors were quantized to int32. 

## Additional Details 

Mobilint use onnx weights and in-house framework for quantization. 

## Reference 

Integer Quantization for Deep Learning Inference: Principles and Empirical Evaluation, Wu et al. (2020), [https://arxiv.org/pdf/2004.09602.pdf](https://arxiv.org/pdf/2004.09602.pdf) 
