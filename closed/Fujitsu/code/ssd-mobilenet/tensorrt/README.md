# SSD-MobileNet Benchmark

## General Information

:warning: **IMPORTANT**: Please use [closed/Fujitsu](closed/Fujitsu) as the working directory when
running the below commands. :warning:

### Goal of this Benchmark

This benchmark performs object detection using SSD-MobileNet network.

### Downloading the dataset

The dataset used for this benchmark is [COCO 2017 validation set](http://images.cocodataset.org/zips/val2017.zip). You can run `bash code/ssd-mobilenet/tensorrt/download_data.sh` to download the dataset.

### Preprocessing data

The input images are in INT8 NCHW or NC/4HW4 format. Please run `python3 code/ssd-mobilenet/tensorrt/preprocess_data.py` to run the preprocessing.

### Model Source

The TensorFlow frozen graph [ssd_mobilenet_v1_coco_2018_01_28.pb](ssd_mobilenet_v1_coco_2018_01_28.pb) is downloaded from the [zenodo link](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz) provided by the [MLPerf inference repository](https://github.com/mlperf/inference/tree/master/vision/classification_and_detection). We convert the TensorFlow frozen graph to UFF format, and then use TensorRT UFF parser with post-processing steps to convert the UFF model to TensorRT network. Details can be found in [SSDMobileNet.py](SSDMobileNet.py).

You can download the model by running `bash code/ssd-mobilenet/tensorrt/download_model.sh`.

## Optimizations

### Plugins

The following TensorRT plugin is used to optimize SSDMobileNet benchmark:
- `NMS_OPT_TRT`: optimizes non-maximum suppression operation
The source codes of this plugin can be found in [../../plugin](../../plugin).

### Lower Precision

To further optimize performance, with minimal impact on detection accuracy, we run the computations in INT8 precision.

### Replace ReLU6 with ReLU

On DLA, we replace ReLU6 with ReLU to achieve further performance.

## Instructions for Audits

### Run Inference through LoadGen

Run the following commands from within the container to run inference through LoadGen:

```
make run RUN_ARGS="--benchmarks=ssd-mobilenet --scenarios=<SCENARIO> --test_mode=PerformanceOnly"
make run RUN_ARGS="--benchmarks=ssd-mobilenet --scenarios=<SCENARIO> --test_mode=AccuracyOnly"
```

To run inference through [Triton Inference Server](https://github.com/triton-inference-server/server) and LoadGen:

```
make run RUN_ARGS="--benchmarks=ssd-mobilenet --scenarios=<SCENARIO> --config_ver=triton --test_mode=PerformanceOnly"
make run RUN_ARGS="--benchmarks=ssd-mobilenet --scenarios=<SCENARIO> --config_ver=triton --test_mode=AccuracyOnly"
```

The performance and the accuracy results will be printed to stdout, and the LoadGen logs can be found in `build/logs`.

### Run with New Weights

Follow these steps to run inference with new weights:

1. Replace `build/models/SSDMobileNet/frozen_inference_graph.pb` with new TensorFlow frozen graph.
2. Run `make calibrate RUN_ARGS="--benchmarks=ssd-mobilenet"` to generate a new calibration cache.
3. Run inference by `make run RUN_ARGS="--benchmarks=ssd-mobilenet --scenarios=<SCENARIO>"`.

### Run with New Validation Dataset

Follow these steps to run inference with new validation dataset:

1. Put the validation dataset under `build/data/coco/val2017` and the new annotation data under `build/data/coco/annotations`.
2. Modify `data_maps/imagenet/val_map.txt` to contain all the file names of the new validation dataset according to their order in the annotation file.
3. Preprocess data by `python3 code/ssd-mobilenet/tensorrt/preprocess_data.py --val_only`.
4. Run inference by `make run RUN_ARGS="--benchmarks=ssd-mobilenet --scenarios=<SCENARIO>"`.

### Run with New Calibration Dataset

Follow these steps to generate a new calibration cache with new calibration dataset:

1. Put the calibration dataset under `build/data/coco/train2017` and the new annotation data under `build/data/coco/annotations`.
2. Modify `data_maps/imagenet/cal_map.txt` to contain all the file names of the new calibration dataset.
3. Preprocess data by `python3 code/ssd-mobilenet/tensorrt/preprocess_data.py --cal_only`.
4. Run `make calibrate RUN_ARGS="--benchmarks=ssd-mobilenet"` to generate a new calibration cache.
