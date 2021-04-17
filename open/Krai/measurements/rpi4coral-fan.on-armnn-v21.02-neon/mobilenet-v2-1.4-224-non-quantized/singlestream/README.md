# MLPerf Inference v1.0 - Image Classification - ArmNN

## SingleStream

- Set up [`program:image-classification-armnn-tflite-loadgen`](https://github.com/krai/ck-mlperf/blob/master/program/image-classification-armnn-tflite-loadgen/README.md) on your SUT.
- Customize the examples below for your SUT.

### Workloads

- [ResNet50](#resnet50)
- [EfficientNet](#efficientnet)
- [MobileNet-v1](#mobilenet_v1)
- [MobileNet-v2](#mobilenet_v2)
- [MobileNet-v3](#mobilenet_v3)

<a name="resnet50"></a>
### ResNet50

#### Accuracy

##### Neon

```bash
$ ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=armnn-v21.02-neon \
--model=resnet50 --scenario=singlestream --mode=accuracy --dataset_size=50000 \
--verbose --sut=firefly
...
```

##### OpenCL

```bash
$ ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=armnn-v21.02-opencl \
--model=resnet50 --scenario=singlestream --mode=accuracy --dataset_size=50000 \
--verbose --sut=firefly
...
```

#### Performance

##### Neon

```bash
$ ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=armnn-v21.02-neon \
--model=resnet50 --scenario=singlestream --mode=performance --target_latency=400 \
--verbose --sut=firefly
```

##### OpenCL

```bash
$ ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=armnn-v21.02-opencl \
--model=resnet50 --scenario=singlestream --mode=performance --target_latency=400 \
--verbose --sut=firefly
```

#### Compliance

##### Neon

```bash
$ ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=armnn-v21.02-neon \
--model=resnet50 --scenario=singlestream --compliance,=TEST04-A,TEST04-B,TEST05,TEST01 \
--verbose --sut=firefly
```

##### OpenCL

```bash
$ ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=armnn-v21.02-opencl \
--model=resnet50 --scenario=singlestream --compliance,=TEST04-A,TEST04-B,TEST05,TEST01 \
--verbose --sut=firefly
```

<a name="efficientnet"></a>
### EfficientNet

#### Accuracy

##### Neon

```bash
$ ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=armnn-v21.02-neon \
--model:=`ck list_variations misc --query_module_uoa=package --tags=model,tflite,effnet --variation_prefix=lite --separator=:` \
--model_extra_tags,=non-quantized,quantized \
--scenario=singlestream --mode=accuracy --dataset_size=50000 \
--verbose --sut=firefly
```

##### OpenCL

```bash
$ ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=armnn-v21.02-opencl \
--model:=`ck list_variations misc --query_module_uoa=package --tags=model,tflite,effnet --variation_prefix=lite --separator=:` \
--model_extra_tags,=non-quantized,quantized \
--scenario=singlestream --mode=accuracy --dataset_size=50000 \
--verbose --sut=firefly
```

#### Performance

##### Neon

###### Use a uniform target latency

```bash
$ ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=armnn-v21.02-neon \
--model:=`ck list_variations misc --query_module_uoa=package --tags=model,tflite,effnet --variation_prefix=lite --separator=:` \
--model_extra_tags,=non-quantized,quantized \
--scenario=singlestream --mode=performance --target_latency=10 \
--verbose --sut=firefly
```

###### [Estimate target latencies](https://github.com/krai/ck-mlperf/tree/master/program/generate-target-latency)

```bash
$ ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=armnn-v21.02-neon \
--model:=`ck list_variations misc --query_module_uoa=package --tags=model,tflite,effnet --variation_prefix=lite --separator=:` \
--model_extra_tags,=non-quantized,quantized \
--mode=performance --scenario=range_singlestream --max_query_count=256 \
--verbose --sut=firefly

$ $(ck find program:generate-target-latency)/run.py --tags=inference_engine.armnn,inference_engine_version.v21.02 | \
sort | tee $(ck find program:image-classification-armnn-tflite-loadgen)/target_latency.txt

$ ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=armnn-v21.02-neon \
--model:=`ck list_variations misc --query_module_uoa=package --tags=model,tflite,effnet --variation_prefix=lite --separator=:` \
--model_extra_tags,=non-quantized,quantized \
--mode=performance --scenario=singlestream \
--target_latency_file=$(ck find program:image-classification-armnn-tflite-loadgen)/target_latency.txt \
--verbose --sut=firefly
```

##### OpenCL

###### Use a uniform target latency

```bash
$ ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=armnn-v21.02-opencl \
--model:=`ck list_variations misc --query_module_uoa=package --tags=model,tflite,effnet --variation_prefix=lite --separator=:` \
--model_extra_tags,=non-quantized,quantized \
--scenario=singlestream --mode=performance --target_latency=10 \
--verbose --sut=firefly
```

###### [Estimate target latencies](https://github.com/krai/ck-mlperf/tree/master/program/generate-target-latency)

```bash
$ ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=armnn-v21.02-opencl \
--model:=`ck list_variations misc --query_module_uoa=package --tags=model,tflite,effnet --variation_prefix=lite --separator=:` \
--model_extra_tags,=non-quantized,quantized \
--mode=performance --scenario=range_singlestream --max_query_count=256 \
--verbose --sut=firefly

$ $(ck find program:generate-target-latency)/run.py --tags=inference_engine.armnn,inference_engine_version.v21.02 | \
sort | tee $(ck find program:image-classification-armnn-tflite-loadgen)/target_latency.txt

$ ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=armnn-v21.02-opencl \
--model:=`ck list_variations misc --query_module_uoa=package --tags=model,tflite,effnet --variation_prefix=lite --separator=:` \
--model_extra_tags,=non-quantized,quantized \
--mode=performance --scenario=singlestream \
--target_latency_file=$(ck find program:image-classification-armnn-tflite-loadgen)/target_latency.txt \
--verbose --sut=firefly
```


<a name="mobilenet_v1"></a>
### MobileNet-v1

#### Accuracy

##### Neon

```bash
$ ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=armnn-v21.02-neon \
--model:=`ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v1 --variation_prefix=v1- --separator=:` \
--model_extra_tags,=non-quantized,quantized \
--scenario=singlestream --mode=accuracy --dataset_size=50000 \
--verbose --sut=firefly
```

##### OpenCL

```bash
$ ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=armnn-v21.02-opencl \
--model:=`ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v1 --variation_prefix=v1- --separator=:` \
--model_extra_tags,=non-quantized,quantized \
--scenario=singlestream --mode=accuracy --dataset_size=50000 \
--verbose --sut=firefly
```

#### Performance

##### Neon

###### Use a uniform target latency

```bash
$ ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=armnn-v21.02-neon \
--model:=`ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v1 --variation_prefix=v1- --separator=:` \
--model_extra_tags,=non-quantized,quantized \
--scenario=singlestream --mode=performance --target_latency=5 \
--verbose --sut=firefly
```

##### [Estimate target latencies](https://github.com/krai/ck-mlperf/tree/master/program/generate-target-latency)

```bash
$ ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=armnn-v21.02-neon \
--model:=`ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v1 --variation_prefix=v1- --separator=:` \
--model_extra_tags,=non-quantized,quantized \
--mode=performance --scenario=range_singlestream --max_query_count=256 \
--verbose --sut=firefly

$ $(ck find program:generate-target-latency)/run.py --tags=inference_engine.armnn,inference_engine_version.v21.02 | \
sort | tee $(ck find program:image-classification-armnn-tflite-loadgen)/target_latency.txt

$ ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=armnn-v21.02-neon \
--model:=`ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v1 --variation_prefix=v1- --separator=:` \
--model_extra_tags,=non-quantized,quantized \
--mode=performance --scenario=singlestream \
--target_latency_file=$(ck find program:image-classification-armnn-tflite-loadgen)/target_latency.txt \
--verbose --sut=firefly
```

##### OpenCL

###### Use a uniform target latency

```bash
$ ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=armnn-v21.02-opencl \
--model:=`ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v1 --variation_prefix=v1- --separator=:` \
--model_extra_tags,=non-quantized,quantized \
--scenario=singlestream --mode=performance --target_latency=5 \
--verbose --sut=firefly
```

###### [Estimate target latencies](https://github.com/krai/ck-mlperf/tree/master/program/generate-target-latency)

```bash
$ ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=armnn-v21.02-opencl \
--model:=`ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v1 --variation_prefix=v1- --separator=:` \
--model_extra_tags,=non-quantized,quantized \
--mode=performance --scenario=range_singlestream --max_query_count=256 \
--verbose --sut=firefly

$ $(ck find program:generate-target-latency)/run.py --tags=inference_engine.armnn,inference_engine_version.v21.02 | \
sort | tee $(ck find program:image-classification-armnn-tflite-loadgen)/target_latency.txt

$ ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=armnn-v21.02-opencl \
--model:=`ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v1 --variation_prefix=v1- --separator=:` \
--model_extra_tags,=non-quantized,quantized \
--mode=performance --scenario=singlestream \
--target_latency_file=$(ck find program:image-classification-armnn-tflite-loadgen)/target_latency.txt \
--verbose --sut=firefly
```

<a name="mobilenet_v2"></a>
### MobileNet-v2

#### Accuracy

##### Neon

```bash
$ ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=armnn-v21.02-neon \
--model:=`ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v2 --variation_prefix=v2- --separator=:` \
--model_extra_tags,=non-quantized,quantized \
--scenario=singlestream --mode=accuracy --dataset_size=50000 \
--verbose --sut=firefly
```

##### OpenCL

```bash
$ ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=armnn-v21.02-opencl \
--model:=`ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v2 --variation_prefix=v2- --separator=:` \
--model_extra_tags,=non-quantized,quantized \
--scenario=singlestream --mode=accuracy --dataset_size=50000 \
--verbose --sut=firefly
```

#### Performance

##### Neon

###### Use a uniform target latency

```bash
$ ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=armnn-v21.02-neon \
--model:=`ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v2 --variation_prefix=v2- --separator=:` \
--model_extra_tags,=non-quantized,quantized \
--scenario=singlestream --mode=performance --target_latency=5 \
--verbose --sut=firefly
```

###### [Estimate target latencies](https://github.com/krai/ck-mlperf/tree/master/program/generate-target-latency)

```bash
$ ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=armnn-v21.02-neon \
--model:=`ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v2 --variation_prefix=v2- --separator=:` \
--model_extra_tags,=non-quantized,quantized \
--mode=performance --scenario=range_singlestream --max_query_count=256 \
--verbose --sut=firefly

$ $(ck find program:generate-target-latency)/run.py --tags=inference_engine.armnn,inference_engine_version.v21.02 | \
sort | tee $(ck find program:image-classification-armnn-tflite-loadgen)/target_latency.txt

$ ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=armnn-v21.02-neon \
--model:=`ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v1 --variation_prefix=v1- --separator=:` \
--model_extra_tags,=non-quantized,quantized \
--mode=performance --scenario=singlestream \
--target_latency_file=$(ck find program:image-classification-armnn-tflite-loadgen)/target_latency.txt \
--verbose --sut=firefly
```

##### OpenCL

###### Use a uniform target latency

```bash
$ ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=armnn-v21.02-opencl \
--model:=`ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v2 --variation_prefix=v2- --separator=:` \
--model_extra_tags,=non-quantized,quantized \
--scenario=singlestream --mode=performance --target_latency=5 \
--verbose --sut=firefly
```

###### [Estimate target latencies](https://github.com/krai/ck-mlperf/tree/master/program/generate-target-latency)

```bash
$ ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=armnn-v21.02-opencl \
--model:=`ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v2 --variation_prefix=v2- --separator=:` \
--model_extra_tags,=non-quantized,quantized \
--mode=performance --scenario=range_singlestream --max_query_count=256 \
--verbose --sut=firefly

$ $(ck find program:generate-target-latency)/run.py --tags=inference_engine.armnn,inference_engine_version.v21.02 | \
sort | tee $(ck find program:image-classification-armnn-tflite-loadgen)/target_latency.txt

$ ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=armnn-v21.02-opencl \
--model:=`ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v1 --variation_prefix=v1- --separator=:` \
--model_extra_tags,=non-quantized,quantized \
--mode=performance --scenario=singlestream \
--target_latency_file=$(ck find program:image-classification-armnn-tflite-loadgen)/target_latency.txt \
--verbose --sut=firefly
```

<a name="mobilenet_v3"></a>
### MobileNet-v3

#### Accuracy

##### Neon

```bash
$ ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=armnn-v21.02-neon \
--model:=`ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v3 --variation_prefix=v3- --separator=:` \
--scenario=singlestream --mode=accuracy --dataset_size=50000 \
--verbose --sut=firefly
```

##### OpenCL

```bash
$ ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=armnn-v21.02-opencl \
--model:=`ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v3 --variation_prefix=v3- --separator=:` \
--scenario=singlestream --mode=accuracy --dataset_size=50000 \
--verbose --sut=firefly
```

#### Performance

##### Neon

###### Use a uniform target latency

```bash
$ ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=armnn-v21.02-neon \
--model:=`ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v3 --variation_prefix=v3- --separator=:` \
--scenario=singlestream --mode=performance --target_latency=6 \
--verbose --sut=firefly
```

###### [Estimate target latencies](https://github.com/krai/ck-mlperf/tree/master/program/generate-target-latency)

```bash
$ ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=armnn-v21.02-neon \
--model:=`ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v3 --variation_prefix=v3- --separator=:` \
--mode=performance --scenario=range_singlestream --max_query_count=256 \
--verbose --sut=firefly

$ $(ck find program:generate-target-latency)/run.py --tags=inference_engine.armnn,inference_engine_version.v21.02 | \
sort | tee $(ck find program:image-classification-armnn-tflite-loadgen)/target_latency.txt

$ ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=armnn-v21.02-neon \
--model:=`ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v3 --variation_prefix=v3- --separator=:` \
--mode=performance --scenario=singlestream \
--target_latency_file=$(ck find program:image-classification-armnn-tflite-loadgen)/target_latency.txt \
--verbose --sut=firefly
```

##### OpenCL

###### Use a uniform target latency

```bash
$ ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=armnn-v21.02-opencl \
--model:=`ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v3 --variation_prefix=v3- --separator=:` \
--scenario=singlestream --mode=performance --target_latency=6 \
--verbose --sut=firefly
```

###### [Estimate target latencies](https://github.com/krai/ck-mlperf/tree/master/program/generate-target-latency)

```bash
$ ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=armnn-v21.02-opencl \
--model:=`ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v3 --variation_prefix=v3- --separator=:` \
--mode=performance --scenario=range_singlestream --max_query_count=256 \
--verbose --sut=firefly

$ $(ck find program:generate-target-latency)/run.py --tags=inference_engine.armnn,inference_engine_version.v21.02 | \
sort | tee $(ck find program:image-classification-armnn-tflite-loadgen)/target_latency.txt

$ ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=armnn-v21.02-opencl \
--model:=`ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v3 --variation_prefix=v3- --separator=:` \
--mode=performance --scenario=singlestream \
--target_latency_file=$(ck find program:image-classification-armnn-tflite-loadgen)/target_latency.txt \
--verbose --sut=firefly
```
