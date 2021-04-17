# SSD-ResNet34 Benchmark

## General Information

:warning: **IMPORTANT**: Please use [closed/NVIDIA](closed/NVIDIA) as the working directory when
running the below commands. :warning:

### Goal of this Benchmark

This benchmark performs object detection using SSD-ResNet34 network.

## Dataset

### Downloading the dataset

The dataset used for this benchmark is [COCO 2017 validation set](http://images.cocodataset.org/zips/val2017.zip). You can run `bash code/ssd-resnet34/openvino/download_data.sh` to download the dataset.

### Preprocessing data

The input images are in FP32 format. Please run `python3 code/ssd-resnet34/openvino/preprocess_data.py` to run the preprocessing.

The file structure expected by the harness are as follows:

```
<MLPERF_CPU_SCRATCH_PATH>/preprocessed_data/coco/annotations/
<MLPERF_CPU_SCRATCH_PATH>/preprocessed_data/coco/val2017/SSDResNet34/fp32/
```

## Model

### Generating model binaries and running INT8 calibration

For generating the required INT8 OpenVino model binaries, follow the instructions in the [calibration documentation./](../../../calibration_triton_cpu/OpenVINO/ssd-resnet34/README.md) 

## Instructions for Audits

### Run Inference through LoadGen

Run the following commands from within the container to run inference through LoadGen on CPU using [Triton inference server](https://github.com/triton-inference-server/server):

```
make run_cpu_harness RUN_ARGS="--benchmarks=ssd-resnet34 --scenarios=<SCENARIO> --config_ver=openvino --test_mode=PerformanceOnly"
make run_cpu_harness RUN_ARGS="--benchmarks=ssd-resnet34 --scenarios=<SCENARIO> --config_ver=openvino --test_mode=AccuracyOnly"
```

The performance and the accuracy results will be printed to stdout, and the LoadGen logs can be found in `build/logs`.
