# MLPerf Inference - Image Classification - ArmNN-TFLite with LoadGen

Running this program is similar to running [`armnn-mlperf:program:image-classification-armnn-tflite`](https://github.com/ARM-software/armnn-mlperf/tree/master/program/image-classification-armnn-tflite) as described in the [ArmNN-MLPerf repo](https://github.com/ARM-software/armnn-mlper://github.com/ARM-software/armnn-mlperf), and [`ck-tensorflow:program:image-classification-tflite`](https://github.com/ctuning/ck-tensorflow/tree/master/program/image-classification-tflite) as described in the [MLPerf Inference repo](https://github.com/mlperf/inference/tree/master/v0.5/classification_and_detection/optional_harness_ck/classification/tflite).

## Detect LoadGen Config file
```
$ ck detect soft --tags=config,loadgen,image-classification-armnn-tflite
```

## Run once
```
firefly $ ck benchmark program:image-classification-armnn-tflite-loadgen \
--speed --repetitions=1 --env.USE_NEON=1 \
--env.CK_VERBOSE=1 \
--env.CK_LOADGEN_SCENARIO=SingleStream \
--env.CK_LOADGEN_MODE=PerformanceOnly \
--env.CK_LOADGEN_DATASET_SIZE=1024 \
--env.CK_LOADGEN_BUFFER_SIZE=1024 \
--dep_add_tags.weights=tflite,resnet \
--dep_add_tags.library=armnn,rel.19.08,tflite,neon \
--dep_add_tags.compiler=gcc,v7 \
--dep_add_tags.images=side.224,preprocessed \
--dep_add_tags.loadgen-config-file=image-classification-armnn-tflite \
--dep_add_tags.python=v3 \
--skip_print_timers
...
------------------------------------------------------------
|            LATENCIES (in nanoseconds and fps)            |
------------------------------------------------------------
Number of queries run: 1024
Min latency:                      389050347ns  (2.57036 fps)
Median latency:                   390678334ns  (2.55965 fps)
Average latency:                  390866753ns  (2.55842 fps)
90 percentile latency:            392202776ns  (2.54970 fps)
Max latency:                      409035981ns  (2.44477 fps)
------------------------------------------------------------

firefly $ ck benchmark program:image-classification-armnn-tflite-loadgen \
--speed --repetitions=1 --env.USE_OPENCL=1 \
--env.CK_VERBOSE=1 \
--env.CK_LOADGEN_SCENARIO=SingleStream \
--env.CK_LOADGEN_MODE=PerformanceOnly \
--env.CK_LOADGEN_DATASET_SIZE=1024 \
--env.CK_LOADGEN_BUFFER_SIZE=1024 \
--dep_add_tags.weights=tflite,resnet \
--dep_add_tags.library=armnn,rel.19.08,tflite,opencl \
--dep_add_tags.compiler=gcc,v7 \
--dep_add_tags.images=side.224,preprocessed \
--dep_add_tags.loadgen-config-file=image-classification-armnn-tflite \
--dep_add_tags.python=v3 \
--skip_print_timers
...
------------------------------------------------------------
|            LATENCIES (in nanoseconds and fps)            |
------------------------------------------------------------
Number of queries run: 1024
Min latency:                      435802872ns  (2.29462 fps)
Median latency:                   439844466ns  (2.27353 fps)
Average latency:                  442231513ns  (2.26126 fps)
90 percentile latency:            447810153ns  (2.23309 fps)
Max latency:                      463122016ns  (2.15926 fps)
------------------------------------------------------------
```

## Explore different models

See `run.sh` scripts we used to generate the [MLPerf Inference v0.5 results](https://mlperf.org/inference-results/) for more details:
```
$ ck list ck-mlperf:script:mlperf-inference-v0.5.*.image-classification
mlperf-inference-v0.5.closed.image-classification
mlperf-inference-v0.5.open.image-classification
```
