# ResNet50 Benchmark

This benchmark performs image classifications using the [ResNet-50](https://arxiv.org/abs/1512.03385) network and the ImageNet dataset.

:warning: **IMPORTANT**: Please use [closed/NVIDIA](closed/NVIDIA) as the working directory when
running the below commands. :warning:

## Dataset

### Downloading / obtaining the dataset

The dataset used for this benchmark is [ImageNet 2012 validation set](http://www.image-net.org/challenges/LSVRC/2012/). Please manually download the dataset and unzip the images to `$MLPERF_SCRATCH_PATH/data/imagenet/`. You can run `bash code/resnet50/tensorrt/download_data.sh` to verify if the images are in the expected locations.

### Preprocessing the dataset for usage

To process the input images to INT8 NCHW format, please run `python3 code/resnet50/tensorrt/preprocess_data.py`. The preprocessed data will be saved to `$MLPERF_SCRATCH_PATH/preprocessed_data/imagenet/ResNet50`.

## Model

### Downloading / obtaining the model

The ONNX model *resnet50_v1.onnx* is downloaded from the [zenodo link](https://zenodo.org/record/2592612/files/resnet50_v1.onnx) provided by the [MLPerf inference repository](https://github.com/mlperf/inference/tree/master/vision/classification_and_detection).

This can also be downloaded by running `bash code/resnet50/tensorrt/download_model.sh`.

### Optimizations

#### Plugins

The following TensorRT plugins are used to optimize ResNet50 benchmark:
- `RES2_FULL_FUSION`: fuses all the res2* layers into one CUDA kernel (for non-Xavier systems)
- `RnRes2Br1Br2c_TRT` version 2: fuses res2a_br1 and res2a_br2c layers into one CUDA kernel (for Xavier systems)
- `RnRes2Br2bBr2c_TRT` version 2: fuses res2b_br2b and res2b_br2c or res2c_br2b and res2c_br2c layers into one CUDA kernel (for Xavier systems)
This plugins are available in [TensorRT 7.2](https://developer.nvidia.com/tensorrt) release.

#### Lower Precision

To further optimize performance, with minimal impact on classification accuracy, we run the computations in INT8 precision.

#### Removal of Softmax

Softmax layer is removed since it does not affect the predicted label.

### Calibration

ResNet50 INT8 is calibrated on a subset of the ImageNet validation set. The indices of this subset can be found at
`data_maps/imagenet/cal_map.txt`. We use TensorRT symmetric calibration, and store the scaling factors in
`code/resnet50/tensorrt/calibrator.cache`.

## Instructions for Audits

### Run Inference through LoadGen

Run the following commands from within the container to run inference through LoadGen:

```
make run RUN_ARGS="--benchmarks=resnet50 --scenarios=<SCENARIO> --test_mode=PerformanceOnly"
make run RUN_ARGS="--benchmarks=resnet50 --scenarios=<SCENARIO> --test_mode=AccuracyOnly"
```

To run inference through [Triton Inference Server](https://github.com/triton-inference-server/server) and LoadGen:

```
make run RUN_ARGS="--benchmarks=resnet50 --scenarios=<SCENARIO> --config_ver=triton --test_mode=PerformanceOnly"
make run RUN_ARGS="--benchmarks=resnet50 --scenarios=<SCENARIO> --config_ver=triton --test_mode=AccuracyOnly"
```

The performance and the accuracy results will be printed to stdout, and the LoadGen logs can be found in `build/logs`.

### Run with New Weights

Follow these steps to run inference with new weights:

1. If the new weights are in TensorFlow frozen graph format, please use [resnet50-to-onnx.sh](https://github.com/mlperf/inference/blob/master/vision/classification_and_detection/tools/resnet50-to-onnx.sh) in the official MLPerf repository to convert it to ONNX format.
2. Replace `build/models/ResNet50/resnet50_v1.onnx` with new ONNX model.
3. Run `make calibrate RUN_ARGS="--benchmarks=resnet50"` to generate a new calibration cache.
4. Run inference by `make run RUN_ARGS="--benchmarks=resnet50 --scenarios=<SCENARIO>"`.

### Run with New Validation Dataset

Follow these steps to run inference with new validation dataset:

1. Put the validation dataset under `build/data/imagenet`.
2. Modify `data_maps/imagenet/val_map.txt` to contain all the file names and the corresponding labels of the new validation dataset.
3. Preprocess data by `python3 code/resnet50/tensorrt/preprocess_data.py --val_only`.
4. Run inference by `make run RUN_ARGS="--benchmarks=resnet50 --scenarios=<SCENARIO>"`.

### Run with New Calibration Dataset

Follow these steps to generate a new calibration cache with new calibration dataset:

1. Put the calibration dataset under `build/data/imagenet`.
2. Modify `data_maps/imagenet/cal_map.txt` to contain all the file names of the new calibration dataset.
3. Preprocess data by `python3 code/resnet50/tensorrt/preprocess_data.py --cal_only`.
4. Run `make calibrate RUN_ARGS="--benchmarks=resnet50"` to generate a new calibration cache.
