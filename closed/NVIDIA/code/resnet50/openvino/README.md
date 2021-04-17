# ResNet50 Benchmark

## General Information

:warning: **IMPORTANT**: Please use [closed/NVIDIA](closed/NVIDIA) as the working directory when
running the below commands. :warning:

### Goal of this Benchmark

This benchmark performs image classifications using the [ResNet-50](https://arxiv.org/abs/1512.03385) network and the ImageNet dataset.

## Dataset

### Downloading the dataset

The dataset used for this benchmark is [ImageNet 2012 validation set](http://www.image-net.org/challenges/LSVRC/2012/). Please manually download the dataset and unzip the images to `$MLPERF_CPU_SCRATCH_PATH/data/imagenet/`. You can run `bash code/resnet50/openvino/download_data.sh` to verify if the images are in the expected locations.

### Preprocessing the dataset
To process the input images to FP32 format used by the Openvino model, please run `python3 code/resnet50/openvino/preprocess_data.py`.

The preprocessed data will be saved to:

```
<MLPERF_CPU_SCRATCH_PATH>/preprocessed_data/imagenet/ResNet50/fp32_nomean
```

## Model

### Generating model binaries and running INT8 calibration

For generating the required INT8 OpenVino model binaries, follow the instructions in the [calibration documentation./](../../../calibration_triton_cpu/OpenVINO/resnet50/README.md) 

## Instructions for Audits

### Run Inference through LoadGen

Run the following commands from within the container to run inference through LoadGen on CPU using [Triton inference server](https://github.com/triton-inference-server/server):

```
make run_cpu_harness RUN_ARGS="--benchmarks=resnet50 --scenarios=<SCENARIO> --config_ver=openvino --test_mode=PerformanceOnly"
make run_cpu_harness RUN_ARGS="--benchmarks=resnet50 --scenarios=<SCENARIO> --config_ver=openvino --test_mode=AccuracyOnly"
```

The performance and the accuracy results will be printed to stdout, and the LoadGen logs can be found in `build/logs`.

### Run with New Validation Dataset

Follow these steps to run inference with new validation dataset:

1. Put the validation dataset under `build/data/imagenet`.
2. Modify `data_maps/imagenet/val_map.txt` to contain all the file names and the corresponding labels of the new validation dataset.
3. Preprocess data by `python3 code/resnet50/openvino/preprocess_data.py`.
4. Run inference by `make run_cpu_harness RUN_ARGS="--benchmarks=resnet50 --config_ver=openvino --scenarios=<SCENARIO>"`.
