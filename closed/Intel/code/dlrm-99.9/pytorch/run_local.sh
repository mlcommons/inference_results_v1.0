#!/bin/bash

source ./run_common.sh

common_opt="--config ./mlperf.conf"
OUTPUT_DIR=`pwd`/output/$name
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

set -x # echo the next command

profiling=0
if [ $profiling == 1 ]; then
    EXTRA_OPS="$EXTRA_OPS --enable-profiling=True"
fi

## multi-instance
if [ $mode == "Server" ] ; then
    python -u python/server.py --profile $profile $common_opt --model $model --model-path $model_path \
           --dataset $dataset --dataset-path $DATA_DIR \
           --output $OUTPUT_DIR $EXTRA_OPS $@
else
    python -u python/offline.py --profile $profile $common_opt --model $model --model-path $model_path \
           --dataset $dataset --dataset-path $DATA_DIR \
           --output $OUTPUT_DIR $EXTRA_OPS $@
fi
