# MLPerf Submission: Mobilint Accelerator
This document describes the details of Mobilint's AI Accelerator for MLPerf Inference v1.0 submission. The details consist of brief information of benchmark systems, system requirements, and lastly step-by-step instructions to run the benchmark.

## Overview
Benchmark for Mobilint Accelerator mainly consists of preprocessor, and System Under Test. Given a compiled model binaries, a user can upload the instructions into the accelerator, upload the preprocessed data, and finally get the result back from the accelerator.

## Benchmark Summary
In this round v1.0, Mobilint submits the benchmark result as below:
| System | Division | Scenario | Model |
| ------ | ----- | ------ | ------ |
| Mobilint Edge | Closed | SingleStream, Offline | Resnet50-v1.5, SSD-MobileNet-v1 |
| Mobilint Edge-Five | Open | Offline | Resnet50-v1.5, SSD-MobileNet-v1 |

For the quality (Accuracy) of the SUT per models are:
| Model | Source model | Measured accuracy |
| ----- | ------ | ------ |
| SSD-MobileNet-v1 | ssd-mobilenet 300x300, ONNX   | 23.094% mAP |
| ResNet50-v1.5 | resnet-50-v1.5, ONNX  | 76.152% |

## System Description
### General System Configurations
* Ubuntu 18.04.4
* Python 3.7.6
### Binaries/Libraries Information
* Xilinx DMA Device driver (xdma.ko)
* Loadgen (branch r1.0; libmlperf_loadgen.a)
  * MD5 Checksum : 17b07168cc1457eb024e870122ab64c2  libmlperf_loadgen.a
* Private acceleration library (libmaccel.so)
  * MD5 Checksum : 79e750943bbe3fec516ff50a4841edfe  libmaccel.so

## Usage
### General Steps
General benchmark steps are following:
1. Prepare the compiled model using the internal compiler. It consists of `imem`, `lmem`, `dmem`, `ddr` binaries per model. Place them into the `out/` folder appropriately, the default location are `out/resnet50` and `out/ssdmobilenet` per model respectively.
2. Preprocess the dataset using `preprocess_resnet.py` or `preprocess_imagenet.py` appropriate for the dataset to be tested.
3. Extract only paths from `instances_val2017.json` or `val_map.txt` using `extract_dataset_list.py`, and place the generated file into the `out/` folder. File name of each model set default for the benchmark are `resnet-dataset.txt` and `ssd-mobilenet-dataset.txt` respectively.
4. Run the SUT (Scenario, Mode, Model, Dataset are the arguments)

```bash
# insmod xdma.ko
$ cp lib/libmaccel.so /usr/lib
$ make
$ python extract_dataset_list.py --dataset-name="DATASET_NAME" --output="OUTPUT_FILENAME" --input="INPUT_FILENAME" --root-path="ROOT_PATH_TO PREPROCESSED_DATA"
$ ./benchmark --scenario="SCENARIO" --mode="MODE" --model="MODEL" --config="mlperf.conf" --config-user="user.conf"
```
### Dataset Path Extractor Arguments
|dataset-name|output|input|root-path|
|-----------|------|-----|------|
|COCO, ImageNet| Output filename | Input file path (`val_map.txt` for ImageNet, `instances_val2017.json` for COCO) | Root path to the preprocessed Image data |

### Benchmark SUT Arguments
|config|config-user|scenario|mode|model|
|------|-----------|--------|----|-----|
|Full path to mlperf.conf|Full path to user.conf|SingleStream, MultiStream, Offline|AccuracyOnly, PerformanceOnly|SSDMobileNetV1, ResNet50|
