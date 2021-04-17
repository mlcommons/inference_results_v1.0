#!/bin/bash

#source ./run_clean.sh

export KMP_BLOCKTIME=1
export OMP_NUM_THREADS=$CPUS_PER_SOCKET
export KMP_AFFINITY="granularity=fine,compact,1,0"
export DNNL_PRIMITIVE_CACHE_CAPACITY=20971520
export LD_PRELOAD="${CONDA_PREFIX}/lib/libtcmalloc.so:${CONDA_PREFIX}/lib/libiomp5.so"
export DLRM_DIR=$PWD/python/model

if [ $# == 1 ]; then
    if [ $1 == "offline" ]; then
        echo "Running offline performance mode"
        ./run_local.sh pytorch dlrm terabyte cpu --scenario Offline --max-ind-range=40000000 --samples-to-aggregate-quantile-file=../tools/dist_quantile.txt --max-batchsize=420000 --samples-per-query-offline=300000 --mlperf-bin-loader
    elif [ $1 == "server" ]; then
        echo "Running sever performance mode"
        ./run_local.sh pytorch dlrm terabyte cpu --scenario Server  --max-ind-range=40000000 --samples-to-aggregate-quantile-file=../tools/dist_quantile.txt --max-sample-size=30 --max-batchsize=9000 --mlperf-bin-loader
    else
        echo "Only offline/server are valid"
    fi
elif [ $# == 2 ]; then
    if [ $1 == "offline" ] && [ $2 == "accuracy" ]; then
        echo "Running offline accuracy mode"
        ./run_local.sh pytorch dlrm terabyte cpu --scenario Offline --max-ind-range=40000000 --samples-to-aggregate-quantile-file=../tools/dist_quantile.txt --max-batchsize=420000 --samples-per-query-offline=300000 --accuracy --mlperf-bin-loader
    elif [ $1 == "server" ] && [ $2 == "accuracy" ]; then
        echo "Running sever accuracy mode"
        ./run_local.sh pytorch dlrm terabyte cpu --scenario Server  --max-ind-range=40000000 --samples-to-aggregate-quantile-file=../tools/dist_quantile.txt --max-sample-size=30  --max-batchsize=9000 --accuracy --mlperf-bin-loader
    else
        echo "Only offline/server accuray are valid"
    fi
else
    echo "Only 1/2 parameters are valid"

fi
