# MLPerf Inference v1.0 Edgecortix-Optimized Implementations for Open Division

This folder contains Edgecortix's optimized implementations for [MLPerf Inference Benchmark v1.0](https://www.mlperf.org/inference-overview/) open division.

## Benchmarks

The following *benchmarks* are part of our submission for MLPerf Inference v1.0:

 - **MobileNet-V2** (mobilenetv2)

## Scenarios

Each of the above benchmarks can be run in one or more of the following two inference *scenarios*:

 - **SingleStream**
 - **Offline**

Please refer to the [MLPerf Inference official page](https://www.mlperf.org/inference-overview/) for explanations about the scenarios.

## Edgecortix Submissions

Our MLPerf Inference v1.0 implementation has the following submissions:

| Benchmark     | Edge Submissions                                                                 |
|---------------|----------------------------------------------------------------------------------|
| MobileNet-V2  | Scenarios: Offline, SingleStream                                                 |


## Edgecortix Submission Systems

The systems that Edgecortix supports and has tested are:

 - Edge systems
   - Dynamic Neural Accelerator (DNA)

## Usage

Here we show an usage example of how to run benchmarks in both performance and accuracy mode.
The whole deployment process:

 - download the FP32 precision model
 - preparation for quantization
 - calibration process
 - quantized model compilation and deployment
 - and finally, model execution

is all part of our additional backend, available [here](code/mobilenetv2/SingleStream/python/backend_edgecortix.py).

This additional backend allows us to use the reference implementation inference scripts provided by MLPerf as usual. It is self-contained and reflects the whole deployment process. For convenience, a simple Python interface that allows executing an already-deployed model is also provided [here](code/mobilenetv2/SingleStream/python/ip_runtime/ip_rt.py).

### Prepare the dataset

The validation dataset directory can be set using the environment variable `DATA_DIR`, in this example it points to the directory where the ImageNet validation dataset is located.

### Performance-only benchmark

First parameter `edgecortix` makes reference to the new backend provided by Edgecortix.
To run the benchmark in performance mode using, for example, the MobileNet-v2 PyTorch model provided by TorchVision package, we should specify `mobilenetv2` as the second parameter.
An additional parameter is the file that contains the list of images used during the calibration stage. In our case, we choose the list number one provided by MLPerf.

```bash
cd vision/classification_and_detection
DATA_DIR=/opt/edgecortix/imagenet/ ./run_local.sh edgecortix mobilenetv2 --dataset-calibration-list ../../calibration/ImageNet/cal_image_list_option_1.txt
```
### Accuracy benchmark

Similarly, to run in accuracy mode:

```bash
cd vision/classification_and_detection
DATA_DIR=/opt/edgecortix/imagenet/ ./run_local.sh edgecortix mobilenetv2 --dataset-calibration-list ../../calibration/ImageNet/cal_image_list_option_1.txt --accuracy
```

## Quantization and calibration

The TorchVision model MobileNet-V2 with FP32 precision weights has been quantized and calibrated using PyTorch's built-in post-training static quantization framework.
Weights are quantized per-channel to 8 bit precision `int8_t`, and activations to 8 bit precision `uint8_t`, using the `FBGemm` quantization back-end. 

Models has been quantized using this script [calibrate_torchvision_model.py](code/mobilenetv2/SingleStream/python/calibrate_torchvision_model.py).

For more information about Pytorch quantization please refer to this [document](https://pytorch.org/blog/introduction-to-quantization-on-pytorch/#post-training-static-quantization) for a more detailed explanation.
