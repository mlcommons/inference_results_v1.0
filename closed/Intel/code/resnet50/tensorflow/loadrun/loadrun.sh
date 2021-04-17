#~/bin/bash

set -x

batchsize_i=$1
iteration_i=$2
scenario=$3
target_qps=$4
spec_override=$5
topology=$6
measure_mode=$7

# default is LBF + no cond
sort_samples=false
schedule_local_batch_first=true
sync_by_cond=false

# default is no heterogeneous setting
loadrun_proc_num=56
loadrun_proc_list=56-111
loadrun_instance_loadgen_makeup=0

if [ "${LOADGEN_MAKEUP}" != "0" ] && [ -n "${LOADGEN_MAKEUP}" ] ; then
  loadrun_instance_loadgen_makeup=${LOADGEN_MAKEUP}
  loadrun_proc_num=4
  loadrun_proc_list=0-3
  echo "instance_loadgen_makeup ${LOADGEN_MAKEUP}"
  echo "loadrun_proc_num ${loadrun_proc_num}"    
  echo "loadrun_proc_list ${loadrun_proc_list}"  
fi

nsockets=$( lscpu | grep 'Socket(s)' | cut -d: -f2 | xargs echo -n)
ncores_per_socket=${ncores_per_socket:=$( lscpu | grep 'Core(s) per socket' | cut -d: -f2 | xargs echo -n )}

logic_cores_0=$(( $nsockets * $ncores_per_socket ))

export OMP_NUM_THREADS=$ncores_per_socket 

export KMP_HW_SUBSET=1t

export KMP_AFFINITY=granularity=fine,compact,1,0

export U8_INPUT_OPT=1

if [ "${SCHEDULER}" = "LBF" ]; then
  schedule_local_batch_first=true
  sort_samples=false
  echo "schedule_local_batch_first true"
  echo "sort_samples false"  
fi

if [ "${SCHEDULER}" = "RR" ]; then
  schedule_local_batch_first=false
  sort_samples=true
  echo "schedule_local_batch_first false"
  echo "sort_samples true"   
fi

if [ "${SYNC_BY_COND}" = "true" ]; then
  sync_by_cond=true
  echo "sync_by_cond true" 
fi

case $spec_override in
  debug)
    echo "debug override spec"
    min_server_query_count=1000
    min_offline_query_count=1000    
    min_duration_ms=1
    min_singlestream_query_count=10
    total_samples=$[$2*$1]
    ;;
  spec)
    echo "use spec"
    total_samples=50000
    ;;    
  *)
    echo "unknown spec override"
    exit 1
    ;;
esac

case $scenario in
  offline)
    echo "scenario offline"
    loadrun_settings="--scenario Offline \
                      --sort_samples ${sort_samples} \
                      --schedule_local_batch_first ${schedule_local_batch_first} \
                      --sync_by_cond true $@"
    echo ${loadrun_settings}
    ;;
  server)
    echo "scenario server"
    loadrun_settings="--scenario Server \
                     --schedule_local_batch_first false \
                     --sync_by_cond false \
                     --sort_samples false $@"
    echo ${loadrun_settings}
    ;;
  singlestream)
    echo -n "scenario singlestream"
    loadrun_settings="--schedule_local_batch_first false \
                     --sync_by_cond false \
                     --sort_samples false \
                     --scenario SingleStream \                     
                     $@" 
    echo ${loadrun_settings}
    ;;    
  *)
    echo -n "unknown scenarion"
    exit 1
    ;;
esac
  
numactl -C $logic_cores_0-$(( $logic_cores_0 + $ncores_per_socket - 1 )) --localalloc ./loadrun  --w 5 --quantized true --batch_size ${batchsize_i} --iterations ${iteration_i} \
      --images "/home/amin/data-workloads/image-net" \
      --labels "/home/amin/data-workloads/image-net/val_map.txt" \
      --init_net_path ../models/$topology/int8_resnet50_v1.pb \
      --performance_samples ${total_samples} \
      --total_samples ${total_samples} \
      --mode ${measure_mode} \
      --include_accuracy true \
      --offline_single_response false \
      --instance_loadgen_makeup ${loadrun_instance_loadgen_makeup} \
      --data_order NHWC \
      --model_name ${topology} \
      --mlperf_config_file_name mlperf.conf \
      --user_config_file_name user.conf \
      ${loadrun_settings}
