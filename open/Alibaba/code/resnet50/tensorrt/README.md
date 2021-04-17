# ResNet50 Benchmark

This benchmark performs image classifications using Alibaba's AutoSinian trained and optimized NAS network and the ImageNet dataset.
The base network is from "<em>Once for All: Train One Network and Specialize it for Efficient Deployment</em>"(https://arxiv.org/pdf/1908.09791.pdf). Autosinian revises the number of channels and retrains a new base network, then searches a sub network for the best performance while meet the accuracy requirement of the Resnet50 Benchmark.

:warning: **IMPORTANT**: Please use [open/Alibaba](open/Alibaba) as the working directory when
running the below commands. :warning:

## Dataset

### Downloading / obtaining the dataset

The dataset used for this benchmark is [ImageNet 2012 validation set](http://www.image-net.org/challenges/LSVRC/2012/). Please manually download the dataset and unzip the images to `$MLPERF_SCRATCH_PATH/data/imagenet/`. You can run `bash code/resnet50/tensorrt/download_data.sh` to verify if the images are in the expected locations.

### Preprocessing the dataset for usage

To process the input images to INT8 NHW4 format, please run `python3 code/resnet50/tensorrt/preprocess_data.py`. The preprocessed data will be saved to `$MLPERF_SCRATCH_PATH/preprocessed_data/imagenet/ResNet50`.

## Model
The ONNX model *ofa_autosinian_is176.onnx* is provided under the code/resnet50/tensorrt/ directory.

### Optimizations

#### Plugins

The `AutoSinianCNN_TRT` plugins is used for ofa_autosinian network, it optimizes the convolution layers in the network.
The plugin is provided as .so files under the code/resnet50/tensorrt/ directory.

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

The performance and the accuracy results will be printed to stdout, and the LoadGen logs can be found in `build/logs`.

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
