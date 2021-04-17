#!/bin/bash

export OMP_NUM_THREADS=28
export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0
#export KMP_AFFINITY=granularity=fine,verbose,noduplicates,compact,1,0
#export OMP_WAIT_POLICY=active
#export KMP_BLOCKTIME=infinite

export MXNET_EXEC_BULK_EXEC_INFERENCE=0
export MXNET_EXEC_BULK_EXEC_TRAIN=0
export MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN=0
#export MKLDNN_VERBOSE=1

#numactl --physcpubind=28-55 --membind=1 \
#python run.py \
#    --scenario=Server \
#    --quantized \
#    --batch_size 24 \
#    --mlperf_conf=mlperf.conf \
#    --user_conf=user.conf
#
#

python run_mxnet.py \
    --scenario=Offline \
    --quantized \
    --batch-size 24 \
    --mlperf-conf=mlperf.conf \
    --user-conf=user.conf
