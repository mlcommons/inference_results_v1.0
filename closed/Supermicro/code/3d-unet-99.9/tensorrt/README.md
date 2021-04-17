# 3D-Unet Benchmark Setup and Usage

This benchmark performs medical image segmentation using 3D-Unet network.

:warning: **IMPORTANT**: Please use [closed/NVIDIA](closed/NVIDIA) as the working directory when
running the below commands. :warning:

## Dataset

### Downloading / obtaining the dataset

The dataset used for this benchmark is [BraTS 2019 Training set](https://www.med.upenn.edu/cbica/brats2019/registration.html). Please download the dataset and unzip the data to `$MLPERF_SCRATCH_PATH/data/BraTS/MICCAI_BraTS_2019_Data_Training`. You can run `bash code/3d-unet/tensorrt/download_data.sh` to verify if the images are in the expected locations.

### Preprocessing the dataset for usage

The preprocessing steps can be split into two parts: the standard preprocessing steps provided by [nnUNet](https://github.com/MIC-DKFZ/nnUNet) followed by converting the images into npy formats in FP16 NDHWC8 format and in INT8 NC/32DHW32 format. Please run `python3 code/3d-unet/tensorrt/preprocess_data.py` to run the preprocessing steps. Note that the preprocessing step requires that the model has been downloaded first.

## Model

### Downloading / obtaining the model

The ONNX model `3dUNetBraTS.onnx` is downloaded from the Zenodo links provided by the [MLPerf inference repository](https://github.com/mlperf/inference/tree/master/vision/medical_imaging/3d-unet). We construct TensorRT network by reading layer and weight information from the ONNX model. Details can be found in [3d-unet.py](3d-unet.py). You can download these models by running `bash code/3d-unet/tensorrt/download_model.sh`.

## Optimizations

### Plugins

The following TensorRT plugins were used to optimize 3D-Unet benchmark:
- `INSTNORM3D_TRT`: optimizes fused 3D InstanceNorm and LeakyReLU operations.
- `PIXELSHUFFLE3D_TRT`: optimizes 3D [PixelShuffle](https://pytorch.org/docs/stable/generated/torch.nn.PixelShuffle.html) (or equivalently the 3D variant of [DepthToSpace](https://github.com/onnx/onnx/blob/master/docs/Operators.md#DepthToSpace)) operation.
The source codes of the plugins can be found in [../../plugin](../../plugin).

### Lower Precision

To further optimize performance, with minimal impact on segmentation accuracy, we run the computations in INT8 precision. We found that INT8 precision satisfies both 99% and 99.9% of the reference FP32 accuracy targets.

### TransposedConvolution -> Convolution + PixelShuffle Conversion

To further optimize performance, we replaced the 2x2x2 stride-2 TransposedConvolution layers in the ONNX graph with and 1x1x1 3D Convolution with 8 times of number of output channels, followed by a 3D [PixelShuffle](https://pytorch.org/docs/stable/generated/torch.nn.PixelShuffle.html) operation. The combination of the Convolution and the PixelShuffle operation is mathematically equivalent to a TransposedConvolution. The replacement is done with the [ONNX Graphsurgeon](https://github.com/NVIDIA/TensorRT/tree/master/tools/onnx-graphsurgeon) tool before the ONNX graph is loaded into the TensorRT ONNX parser.

## Instructions for Auditors

### Run Inference through LoadGen

Run the following commands from within the container to run inference through LoadGen:

```
make run RUN_ARGS="--benchmarks=3d-unet --scenarios=<SCENARIO> --config_ver=default --test_mode=PerformanceOnly"
make run RUN_ARGS="--benchmarks=3d-unet --scenarios=<SCENARIO> --config_ver=default --test_mode=AccuracyOnly"
make run RUN_ARGS="--benchmarks=3d-unet --scenarios=<SCENARIO> --config_ver=high_accuracy --test_mode=PerformanceOnly"
make run RUN_ARGS="--benchmarks=3d-unet --scenarios=<SCENARIO> --config_ver=high_accuracy --test_mode=AccuracyOnly"
```

To run inference through [Triton Inference Server](https://github.com/triton-inference-server/server) and LoadGen:

```
make run RUN_ARGS="--benchmarks=3d-unet --scenarios=<SCENARIO> --config_ver=triton --test_mode=PerformanceOnly"
make run RUN_ARGS="--benchmarks=3d-unet --scenarios=<SCENARIO> --config_ver=triton --test_mode=AccuracyOnly"
make run RUN_ARGS="--benchmarks=3d-unet --scenarios=<SCENARIO> --config_ver=high_accuracy_triton --test_mode=PerformanceOnly"
make run RUN_ARGS="--benchmarks=3d-unet --scenarios=<SCENARIO> --config_ver=high_accuracy_triton --test_mode=AccuracyOnly"
```

The performance and the accuracy results will be printed to stdout, and the LoadGen logs can be found in `build/logs`.

### Run with New Weights

Follow these steps to run inference with new weights:

1. Replace `build/models/3d-unet/3dUNetBraTS.onnx` with new ONNX model.
2. Run `make calibrate RUN_ARGS="--benchmarks=3d-unet"` to generate a new calibration cache.
3. Run inference by `make run RUN_ARGS="--benchmarks=3d-unet --scenarios=<SCENARIO> --config_ver=high_accuracy"`.

### Run with New Validation or Calibration Dataset

Follow these steps to run inference with new validation or calibration dataset:

1. Put the new dataset under `build/data/BraTS/MICCAI_BraTS_2019_Data_Training`.
2. Preprocess data by `python3 code/3d-unet/tensorrt/preprocess_data.py`.
3. Run `make calibrate RUN_ARGS="--benchmarks=3d-unet"` to generate a new calibration cache.
4. Run inference by `make run RUN_ARGS="--benchmarks=3d-unet --scenarios=<SCENARIO> --config_ver=high_accuracy"`.
