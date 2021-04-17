# **Neuchips MLperf Quantization**
Neuchips adopts dynamic-range symmetric quantization to quantize weights, activations and embeddings from FP32 to int8/uint8.

## **Weights and Embeddings**
Neuchips adopts layer-wise quantization for weights and table-wise quantization for embeddings.

* Step1: find the range R of weights/embeddings for each layer/table. 
* Step2: find the minimum number E, which R is in the range [-2<sup>E</sup>, 2<sup>E</sup>). 
* Step3: quantize weights/embeddings to int8 with format Qn.m. If E is larger than 0, n = E+1 and m = 7-E. Otherwise, n = 1 and m = 7-E. 

## **Activations**
Neuchips adopts layer-wise quantization for activations.

* Step1: find the range R of activations for each layer.
* Step2: find the minimum number E, which R is in the range [-2<sup>E</sup>, 2<sup>E</sup>).
* Step3: quantize activations to uint8 with format Qn.m if the activation function is Relu. If E is larger than 0, n = E and m = 8-E. Otherwise, n = 0 and m = 8-E.
         quantize activations to int8 with format Qn.m if the activation function is not Relu. If E is larger than 0, n = E+1 and m = 7-E. Otherwise, n = 1 and m = 7-E.
