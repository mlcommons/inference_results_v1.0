#~/bin/bash
set -x

model=$1
mode=$2

net_conf=${model}

if [ ${model} == "mobilenet" ];then
   net_conf=mobilenet
fi

set -x

#export SCHEDULER=RR

export OMP_NUM_THREADS=28

export KMP_HW_SUBSET=1t
export KMP_AFFINITY=granularity=fine,compact,1,0
#export KMP_BLOCKTIME=0
export TF_XLA_FLAGS="--tf_xla_cpu_global_jit"


if [ ${model} == "mobilenet" ];then
  net_conf="mobilenet"
else
  net_conf="resnet50"
fi

if [ "${model}" == "mobilenet" ];then
echo "run mobilenet ......."
echo "************************************"
numactl -C 0-27 -l ./loadrun --w 200 --quantized true \
      --batch_size 1 \
      --iterations 50000 \
      --images "/localdisk/amin/data-workloads/image-net/" \
      --labels "/localdisk/amin/data-workloads/image-net/val.txt" \
      --init_net_path ../models/${model}/init_net_int8.pb \
      --shared_memory_option USE_LOCAL \
      --shared_weight USE_LOCAL \
      --min_query_count 1024 \
      --min_duration_ms 60000 \
      --performance_samples 50000 \
      --single_stream_expected_latency_ns 500000 \
      --total_samples 50000 \
      --scenario singlestream \
      --net_conf ${net_conf} \
      --schedule_local_batch_first false \
      --include_accuracy false \
      --data_order NHWC \
      --model_name ${model} \
      --mode ${mode}

     # --mlperf_config_file_name mlperf.conf \
     # --user_config_file_name user.conf.${model} \
else
numactl -C 0-27 -l ./loadrun --w 200 --quantized true \
      --batch_size 1 \
      --iterations 50000 \
      --images "/localdisk/amin/data-workloads/image-net/" \
      --labels "/localdisk/amin/data-workloads/image-net/val_map.txt" --init_net_path ../models/${model}/init_net_int8.pb \
      --shared_memory_option USE_LOCAL \
      --shared_weight USE_LOCAL \
      --min_query_count 1024 \
      --min_duration_ms 60000 \
      --performance_samples 50000 \
      --total_samples 50000 \
      --scenario SingleStream \
      --net_conf ${net_conf} \
      --schedule_local_batch_first false \
      --mlperf_config_file_name mlperf.conf \
      --user_config_file_name user.conf.${model} \
      --include_accuracy false \
      --data_order NHWC \
      --model_name ${model} \
      --mode ${mode}
fi 

mkdir -p results/${model}/singlestream/performance
mkdir -p results/${model}/singlestream/accuracy

if [ "${mode}" == "PerformanceOnly" ];then
  cp mlperf_log_detail.txt mlperf_log_summary.txt results/${model}/singlestream/performance
fi

if [ "${mode}" == "AccuracyOnly" ];then
  cp mlperf_log_accuracy.json results/${model}/singlestream/accuracy
fi
