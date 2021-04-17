# Xilinx MLPerf Quantization

We used per-tensor symmetric for both weights and activation and uses an 8-bit integer as its numerical precision.
Moreover, a power-of-2 scaling factor is used in quantization. 
The formula is: Q(x) = clamp(round(x * scale), -128, 127). Here scale = 2 ^ p and p is an integer value.
In post-training quantization, the distributions of weights and activation are used to get the optimal scaling factors.

## Weights

A per-tensor symmetric quantization is used. The scaling factor of each tensor is obtained according to the distribution by minimizing the mean square error of the quantized and float tensor.
Both weight tensors and bias tensors are quantized to int8.

## Activations

A per-tensor symmetric quantization is used. The scaling factor of each tensor is calibrated by invoking the model on the calibration dataset (from the mlperf calibration dataset). The histogram of the scaling factor is record over mini-batches and the most commonly used value is set.
Based on the scaling factor the activation tensor is clamped and quantized.  

## Further improvement

To improve quantization performance we employ cross layer equalization[1] when needed.

## Quantization in Plugins

Xilinx's closed division submissions use our proprietary software stack named Vitis-AI[2], which implements the scheme described above. 

## Open Division

Our Open Division submissions use exactly the same calibration and quantization setting. 

## References
[1]Nagel, Markus, et al. "Data-free quantization through weight equalization and bias correction." Proceedings of the IEEE International Conference on Computer Vision. 2019.<br />
[2]Vitis-AI User Guide, https://www.xilinx.com/support/documentation/sw_manuals/vitis_ai/1_3/ug1414-vitis-ai.pdf, 2021

