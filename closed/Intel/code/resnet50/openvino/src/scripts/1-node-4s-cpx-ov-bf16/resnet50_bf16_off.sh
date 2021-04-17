#!/bin/bash

export OV_MLPERF_BIN=</path/to/ov_mlperf>

${OV_MLPERF_BIN} --scenario Offline \
	--mode Performance \
	--mlperf_conf mlperf.conf \ 
	--user_conf user.conf \ 
	--model_name resnet50 \
	--data_path </path/to/imagenet-dataset> \
	--nireq 224 \
	--nthreads 112 \
	--nstreams 112 \
	--total_sample_count 1024 \
	--warmup_iters 1000 \
	--model_path </path/to/resnet50-fp16.xml> \
	--enforcebf16


