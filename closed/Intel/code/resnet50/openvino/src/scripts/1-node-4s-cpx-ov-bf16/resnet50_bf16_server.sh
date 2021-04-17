#!/bin/bash

export OV_MLPERF_BIN=</path/to/ov_mlperf>

${OV_MLPERF_BIN} --scenario Server \
	--mode Performance \
	--mlperf_conf mlperf.conf \ 
	--user_conf user.conf \ 
	--model_name resnet50 \
	--data_path </path/to/imagenet-dataset> \
	--nireq 56 \
	--nthreads 112 \
	--nstreams 28 \
	--total_sample_count 1024 \
	--warmup_iters 1000 \
	--model_path </path/to/resnet50-fp16.xml> \
	--enforcebf16


