#~/bin/bash

set -ex

# Remove logs
log_files=(mlperf_log_*)
if [[ -f "$log_files" ]]; then
  rm ./mlperf_log_*
fi

batchsize_i=$1
iteration_i=$((50000/$batchsize_i))
scenario=$2
instance_i=$3
topology=$4
total_cores=$5
measure_mode=$6 

export SCHEDULER=RR

shared_weight_switch=shw_off
spec_override=spec

case $shared_weight_switch in
  shw_on)
    echo "shared_weight_switch on"
    shared_weight_create=CREATE_USE_SHM
    shared_weight_use=USE_SHM
    ;;
  shw_off)
    echo -n "shared_weight_switch off"
    shared_weight_create=USE_LOCAL
    shared_weight_use=USE_LOCAL
    ;;
  *)
    echo -n "unknown option of shared_weight_switch"
    exit 1
    ;;
esac

nsockets=$( lscpu | grep 'Socket(s)' | cut -d: -f2 | xargs echo -n)
ncores_per_socket=${ncores_per_socket:=$( lscpu | grep 'Core(s) per socket' | cut -d: -f2 | xargs echo -n)}

if [ ${total_cores} == "" ];then
   total_cores=$(( $ncores_per_socket * $nsockets ))
fi
ncores_per_instance=$((${total_cores} / ${instance_i}))

instances=""
for((i=0;$i<${total_cores};i=$(($i + ${ncores_per_instance}))));  
do
instances=" $instances --instance $i "
done

echo ./loadrun.sh ${batchsize_i} ${iteration_i} ${scenario} ${target_qps} ${spec_override} ${topology} ${measure_mode} $instances --server server99  --loadrun_queue_size ${batchsize_i} 
./loadrun.sh ${batchsize_i} ${iteration_i} ${scenario} 0 ${spec_override} ${topology} ${measure_mode} $instances --server server99  --loadrun_queue_size ${batchsize_i} & 

sleep 6

for((j=0;$j<${total_cores};j=$(($j + ${ncores_per_instance}))));  
do

if [[ $((${j} % ${ncores_per_socket})) -eq 0 ]];
then
   echo ./netrun.sh ${batchsize_i} ${iteration_i} ${j}-$(($j + ${ncores_per_instance} - 1)) ${ncores_per_instance} $j server99 CREATE_USE_SHM $(($j / ${ncores_per_socket})) ${shared_weight_create} ${topology} ${measure_mode} ${scenario}
   ./netrun.sh ${batchsize_i} ${iteration_i} "${j}-$(($j + ${ncores_per_instance} -1))" ${ncores_per_instance} $j server99 CREATE_USE_SHM $(($j / ${ncores_per_socket})) ${shared_weight_create} ${topology} ${measure_mode} ${scenario} &
else
   echo ./netrun.sh ${batchsize_i} ${iteration_i} ${j}-$(($j + ${ncores_per_instance} -1)) ${ncores_per_instance} $j server99 USE_SHM $(($j / ${ncores_per_socket})) ${shared_weight_create} ${topology} ${measure_mode} ${scenario}
   ./netrun.sh ${batchsize_i} ${iteration_i} "${j}-$(($j + ${ncores_per_instance} -1))" ${ncores_per_instance} $j server99 USE_SHM $(($j / ${ncores_per_socket})) ${shared_weight_create} ${topology} ${measure_mode} ${scenario} &
fi

done

wait

mkdir -p results/${topology}/${scenario}/

if [ "${measure_mode}" == "AccuracyOnly" ];then
  if [ -f ./mlperf_log_accuracy.json ];then
    mkdir -p results/${topology}/${scenario}/accuracy
    cp ./mlperf_log_* results/${topology}/${scenario}/accuracy/
  fi
fi

if [ "${measure_mode}" == "PerformanceOnly" ];then
  if [ -f ./mlperf_log_summary.txt ];then
    mkdir -p results/${topology}/${scenario}/performance
    cp ./mlperf_log_* results/${topology}/${scenario}/performance/
  fi
fi

exit
