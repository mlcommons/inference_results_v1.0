# Qualcomm - Calibration Details

We use a combination of several post-training quantization techniques.
(The exact combination of techniques for each workload is to be confirmed.)

## Regular profile-guided quantization techniques

We pass a set of calibration images through the neural network to obtain a
profile of tensor values for the network operations.  We then use the profile
to calculate the scales and offsets for quantization to have a negligible
impact on accuracy.

## Advanced AIMET quantization techniques

AI Model Efficiency Toolkit
([AIMET](https://github.com/quic/aimet/blob/develop/README.md)) is an
open-source library that implements advanced post-training quantization techniques for
pre-trained neural network models. It provides features that have been proven to
improve run-time performance of deep learning neural network models with lower
compute and memory requirements and minimal impact to task accuracy.

### Cross Layer Equalization

In some models, the ranges of weight parameters show a wide variance for
different channels. This technique aims to equalize the parameter ranges across
different channels by scaling the channel weights across consecutive layers.
This helps increase the range for layers with a low range and reduce the range
for layers with a high range. As different channels assume similar ranges, the
same quantization parameters can be used for the weights across all channels.

### Bias Correction

Quantization sometimes leads to a shift layer outputs. This technique helps
correct this shift by adjusting the bias parameters of that layer. The bias
parameter is iteratively corrected for each layer. The layer which bias is to
be corrected, and all layers above it, are quantized.

AIMET supports two Bias Correction techniques:

1. In Empirical Bias Correction, representative data is passed through both the
   floating point model and the quantized model. Outputs are extracted from
both models for the layer to be corrected and then used for correcting the bias
parameter. This process is repeated for all layers in the model.

2. In Analytical Bias Correction, data from Batch Norms is used when available
   instead.

### Further Reading

1. ["Data-Free Quantization Through Weight Equalization and Bias Correction"](https://iccv2019.thecvf.com/). [ICCV 2019](https://iccv2019.thecvf.com/).
2. [AIMET on GitHub Pages](https://quic.github.io/aimet-pages/index.html).
