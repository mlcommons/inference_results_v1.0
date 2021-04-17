#!/bin/bash

export OV_MLPERF_BIN=</path/to/ov_mlperf>

${OV_MLPERF_BIN} --scenario Server \
        --mode Performance \
        --mlperf_conf mlperf.conf \
        --user_conf user.conf \
        --model_name ssd-resnet34 \
        --dataset coco \
        --data_path </path/to/dataset-coco-2017-val> \
        --model_path </path/to/ssd-resnet34_int8.xml> \
        --total_sample_count 1024 \
        --perf_sample_count 64 \
        --nireq 8 \
        --nstreams 4 \
        --nthreads 80 \
        --batch_size 1

