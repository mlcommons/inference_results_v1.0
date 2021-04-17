#!/bin/bash

export OV_MLPERF_BIN=</path/to/ov_mlperf>

${OV_MLPERF_BIN} --scenario Server \
	--mode Accuracy \
	--mlperf_conf mlperf.conf \ 
	--user_conf user.conf \ 
	--model_name resnet50 \
	--dataset imagenet \
	--data_path </path/to/imagenet-dataset> \
	--model_path </path/to/resnet50_int8.xml> \
        --total_sample_count 50000 \
        --perf_sample_count 1024 \
        --nireq 28 \
        --nstreams 28 \
        --nthreads 112 \
        --batch_size 1

