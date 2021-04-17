# 3D-Unet Benchmark

## General Information

:warning: **IMPORTANT**: Please use [closed/NVIDIA](closed/NVIDIA) as the working directory when
running the below commands. :warning:

### Goal of this Benchmark

This benchmark performs medical image segmentation using 3D-Unet network.

## Dataset

### Downloading the dataset

The dataset used for this benchmark is [BraTS 2019 Training set](https://www.med.upenn.edu/cbica/brats2019/registration.html). Please download the dataset and unzip the data to `$MLPERF_CPU_SCRATCH_PATH/data/BraTS/MICCAI_BraTS_2019_Data_Training`. You can run `bash code/3d-unet/openvino/download_data.sh` to verify if the images are in the expected locations.

### Preprocessing the dataset

The preprocessing steps are split into two parts: the standard preprocessing steps provided by [nnUNet](https://github.com/MIC-DKFZ/nnUNet) followed by converting the images into npy formats in FP32 format. First run `bash code/3d-unet/openvino/download_model.sh` to download the model checkpoint necessary to calibrate the data. After, run `python3 code/3d-unet/openvino/preprocess_data.py` to run the preprocessing steps.

The file structure expected by the harness are as follows:

```
<MLPERF_CPU_SCRATCH_PATH>/preprocessed_data/brats/brats_reference_preprocessed/
<MLPERF_CPU_SCRATCH_PATH>/preprocessed_data/brats/brats_npy/fp32/
<MLPERF_CPU_SCRATCH_PATH>/preprocessed_data/brats/brats_reference_raw
```

## Model

### Generating model binaries and running INT8 calibration

For generating the required INT8 OpenVino model binaries, follow the instructions in the [calibration documentation./](../../../calibration_triton_cpu/OpenVINO/3d-unet/README.md) 

## Instructions for Auditors

### Run Inference through LoadGen

Run the following commands from within the container to run inference through LoadGen on CPU using [Triton inference server](https://github.com/triton-inference-server/server):

```
make run_cpu_harness RUN_ARGS="--benchmarks=3d-unet --scenarios=<SCENARIO> --config_ver=openvino --test_mode=PerformanceOnly"
make run_cpu_harness RUN_ARGS="--benchmarks=3d-unet --scenarios=<SCENARIO> --config_ver=openvino --test_mode=AccuracyOnly"
```

The performance and the accuracy results will be printed to stdout, and the LoadGen logs can be found in `build/logs`.
