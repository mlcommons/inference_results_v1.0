# MLPerf Inference - Image Classification - TFLite

This C++ implementation runs TFLite models for Image Classification using TFLite.

## Prerequisites

### [Preprocess ImageNet on an x86 machine](https://github.com/arm-software/armnn-mlperf#preprocess-on-an-x86-machine-and-detect-on-an-arm-dev-board)

#### `model-tflite-mlperf-resnet*`, `model-tflite-mlperf-efficientnet-lite0`, `model-tf-and-tflite-mlperf-mobilenet*` (resolution 224)

```bash
$ ck install package --tags=dataset,imagenet,preprocessed,using-opencv,side.224,full --ask
```

#### `model-tf-and-tflite-mlperf-mobilenet*` (resolution 192)

```bash
$ ck install package --tags=dataset,imagenet,preprocessed,using-opencv,side.192,full --ask
```

#### `model-tf-and-tflite-mlperf-mobilenet*` (resolution 160)

```bash
$ ck install package --tags=dataset,imagenet,preprocessed,using-opencv,side.160,full --ask
```

#### `model-tf-and-tflite-mlperf-mobilenet*` (resolution 128)

```bash
$ ck install package --tags=dataset,imagenet,preprocessed,using-opencv,side.128,full --ask
```

#### `model-tf-and-tflite-mlperf-mobilenet*` (resolution 96)

```bash
$ ck install package --tags=dataset,imagenet,preprocessed,using-opencv,side.96,full --ask
```

#### `model-tflite-mlperf-efficientnet-lite1`

```bash
$ ck install package --tags=dataset,imagenet,preprocessed,using-opencv,side.240,full --ask
```

#### `model-tflite-mlperf-efficientnet-lite2`

```bash
$ ck install package --tags=dataset,imagenet,preprocessed,using-opencv,side.260,full --ask
```

#### `model-tflite-mlperf-efficientnet-lite3`

```bash
$ ck install package --tags=dataset,imagenet,preprocessed,using-opencv,side.280,full --ask
```

#### `model-tflite-mlperf-efficientnet-lite4`

```bash
$ ck install package --tags=dataset,imagenet,preprocessed,using-opencv,side.300,full --ask
```

### [Detect ImageNet on a dev board](https://github.com/arm-software/armnn-mlperf#preprocess-on-an-x86-machine-and-detect-on-an-arm-dev-board)

Copy a preprocessed ImageNet dataset onto a dev board e.g. under `/datasets` and register it with CK according to its resolution e.g.:

```bash
$ echo opencv-side.240 | ck detect soft --tags=dataset,imagenet,preprocessed,rgb8 \
--extra_tags=using-opencv,crop.875,full,inter.linear,side.240 \
--full_path=/datasets/dataset-imagenet-preprocessed-using-opencv-crop.875-full-inter.linear-side.240/ILSVRC2012_val_00000001.rgb8
```


## Run once (classical CK interface)

Running this program is similar to running [`ck-tensorflow:program:image-classification-tflite`](https://github.com/ctuning/ck-tensorflow/tree/master/program/image-classification-tflite),
as described in the [MLPerf Inference repo](https://github.com/mlperf/inference/tree/master/v0.5/classification_and_detection/optional_harness_ck/classification/tflite).

```bash
firefly $ ck benchmark program:image-classification-tflite-loadgen \
--speed --repetitions=1 \
--env.CK_VERBOSE=1 \
--env.CK_LOADGEN_SCENARIO=SingleStream \
--env.CK_LOADGEN_MODE=PerformanceOnly \
--env.CK_LOADGEN_DATASET_SIZE=1024 \
--env.CK_LOADGEN_BUFFER_SIZE=1024 \
--dep_add_tags.weights=model,tflite,resnet \
--dep_add_tags.library=tflite,v1.15 \
--dep_add_tags.compiler=gcc,v7 \
--dep_add_tags.images=side.224,preprocessed \
--dep_add_tags.loadgen-config-file=image-classification-tflite \
--dep_add_tags.python=v3 \
--skip_print_timers
...
------------------------------------------------------------
|            LATENCIES (in nanoseconds and fps)            |
------------------------------------------------------------
Number of queries run: 1024
Min latency:                      397952762ns  (2.51286 fps)
Median latency:                   426440993ns  (2.34499 fps)
Average latency:                  433287227ns  (2.30794 fps)
90 percentile latency:            460194271ns  (2.173 fps)
Max latency:                      679467557ns  (1.47174 fps)
------------------------------------------------------------
```

## Explore different models
**TODO**
