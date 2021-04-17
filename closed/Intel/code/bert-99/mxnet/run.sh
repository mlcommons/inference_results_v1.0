#!/bin/bash
export OMP_NUM_THREADS=$CPUS_PER_INSTANCE
export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0
export MKL_THREADING_LAYER=INTEL
export KMP_STACKSIZE=16M
#export KMP_LIBRARY=turnaround

if [ "x$BATCH_SIZE" == "x" ]; then
    echo "BATCH_SIZE not set" && exit 1
fi
if [ "x$NUM_INSTANCE" == "x" ]; then
    echo "NUM_INSTANCE not set" && exit 1
fi

NUM_PHY_CPUS=$(( $NUM_INSTANCE*$CPUS_PER_INSTANCE ))

if [ $# == 1 ]; then
    if [ $1 == "offline" ]; then
        echo "Running offline performance mode"
        python run_mxnet.py \
        --quantized_model_prefix=converted_from_tf_to_mxnet/offline_model/model_bert_squad_quantized_customize \
        --batch-size=$BATCH_SIZE \
        --num-instance=$NUM_INSTANCE \
        --num-phy-cpus=$NUM_PHY_CPUS \
        --quantized \
        --batching=Adaptive \
        --warmup \
        --mlperf-conf=mlperf.conf \
        --user-conf=user.conf \
        --scenario=Offline

    elif [ $1 == "calibrate" ]; then
        echo "Running offline performance mode"
        python run_mxnet.py \
        --quantized_model_prefix=converted_from_tf_to_mxnet/offline_model/model_bert_squad_quantized_customize \
        --batch-size=$BATCH_SIZE \
        --num-instance=$NUM_INSTANCE \
        --num-phy-cpus=$NUM_PHY_CPUS \
        --quantized \
        --batching=Adaptive \
        --perf_calibrate \
        --mlperf-conf=mlperf.conf \
        --user-conf=user.conf \
        --scenario=Offline

     elif [ $1 == "server" ]; then
        echo "Running sever performance mode"
        MXNET_DISABLE_ONEDNN_BRGEMM_FC=1 python run_mxnet.py  \
        --quantized_model_prefix=converted_from_tf_to_mxnet/server_model/model_bert_squad_quantized_customize \
        --batch-size=$BATCH_SIZE \
        --num-instance=$NUM_INSTANCE \
        --num-phy-cpus=$NUM_PHY_CPUS \
        --quantized \
        --warmup \
        --mlperf-conf=mlperf.conf \
        --user-conf=user.conf \
        --scenario=Server

    else
        echo "Only offline/server are valid"
    fi
elif [ $# == 2 ]; then
    if [ $1 == "offline" ] && [ $2 == "accuracy" ]; then
        echo "Running offline accuracy mode"
        python run_mxnet.py \
        --quantized_model_prefix=converted_from_tf_to_mxnet/offline_model/model_bert_squad_quantized_customize \
        --batch-size=$BATCH_SIZE \
        --num-instance=$NUM_INSTANCE \
        --num-phy-cpus=$NUM_PHY_CPUS \
        --quantized \
        --mlperf-conf=mlperf.conf \
        --user-conf=user.conf \
        --accuracy \
        --scenario=Offline
    elif [ $1 == "server" ] && [ $2 == "accuracy" ]; then
        echo "Running sever accuracy mode"
        MXNET_DISABLE_ONEDNN_BRGEMM_FC=1 python run_mxnet.py \
        --quantized_model_prefix=converted_from_tf_to_mxnet/server_model/model_bert_squad_quantized_customize \
        --batch-size=$BATCH_SIZE \
        --num-instance=$NUM_INSTANCE \
        --num-phy-cpus=$NUM_PHY_CPUS \
        --quantized \
        --warmup \
        --mlperf-conf=mlperf.conf \
        --user-conf=user.conf \
        --accuracy \
        --scenario=Server
    else
        echo "Only offline/server accuray are valid"
    fi
else
    echo "Only 1/2 parameters are valid"
fi
