#~/bin/bash
set -x

#$1 batch size
#$2 iterations
#$3 the processor list
#$4 the OMP thread count
#$5 instance name
#$6 server name
#$7 shared mem option
#$8 numa id
#$9 shared weight option
#$10 model
#$11 measure_mode
#$12 scenario

scenario=${12} 

schedule_local_batch_first=true
sync_by_cond=false

if [ "${SCHEDULER}" = "LBF" ]; then
  schedule_local_batch_first=true
  echo "schedule_local_batch_first true"
fi

if [ "${SCHEDULER}" = "RR" ]; then
  schedule_local_batch_first=false
  echo "schedule_local_batch_first false"
fi

if [ "${SYNC_BY_COND}" = "true" ]; then
  sync_by_cond=true
  echo "sync_by_cond true" 
fi

spec_override=spec

case $spec_override in
  debug)
    echo "debug override spec"
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
    netrun_settings="--schedule_local_batch_first ${schedule_local_batch_first} \
                     --sync_by_cond true"
    echo ${netrun_settings}
    ;;
  server)
    echo "scenario server"
    netrun_settings="--schedule_local_batch_first false \
                     --sync_by_cond false"
    echo ${netrun_settings}
    ;;
  singlestream)
    echo -n "scenario singlestream"
    netrun_settings="--schedule_local_batch_first false \
                     --sync_by_cond false" 
    echo ${netrun_settings}
    ;;
  *)
    echo -n "unknown scenarion"
    exit 1
    ;;
esac


# feel free to use numacrl or taskset to control affinity
export OMP_NUM_THREADS=$4
export KMP_AFFINITY="proclist=[$3],granularity=fine,explicit"
export KMP_BLOCKTIME=0
export TF_XLA_FLAGS="--tf_xla_cpu_global_jit"

export KMP_HW_SUBSET=1t

images_path=""
labels_file=""

imagenet_path="/home/amin/data-workloads/image-net"
imagenet_labels_path="/home/amin/data-workloads/image-net/val_map.txt"

coco_path=""
coco_labels_path=""

if [ ${10} == "resnet50" ];then
  net_conf="resnet50"
  images_path=$imagenet_path
  labels_file=$imagenet_labels_path
elif [ ${10} == "mobilenet" ];then
  net_conf="mobilenet"
  images_path=$imagenet_path
  labels_file=$imagenet_labels_path
elif [ ${10} == "ssd_resnet34" ];then
  net_conf="ssd_resnet34"
  images_path=$coco_path
  labels_file=$coco_labels_path
elif [ ${10} == "ssd_mobilenet" ];then
  net_conf="ssd_mobilenet"
  images_path=$coco_path
  labels_file=$coco_labels_path
else
  echo -n "unknown model"
  exit 1
fi


# batchsize_i * iteration_i decides how many imgs will be loaded to ram
echo numactl -C $3 ./netrun  --w 100 --quantized true --batch_size $1 --iterations $2 \
      --images $images_path \
      --labels $labels_file \
      --init_net_path ../models/${10}/int8_resnet50_v1.pb \
      --random_multibatch true \
      --shared_memory_option $7 \
      --shared_weight $9 \
      --numa_id $8 \
      --instance $5 \
      --server $6 \
      --mode ${11} \
      --include_accuracy true \
      --data_order NHWC \
      --net_conf $net_conf \
      ${netrun_settings}

LD_PRELOAD=./libtcmalloc.so.4.3.0 numactl -C $3 --localalloc ./netrun  --w 100 --quantized true --batch_size $1 --iterations $2 \
      --images $images_path \
      --labels $labels_file \
      --init_net_path ../models/${10}/int8_resnet50_v1.pb \
      --total_samples ${total_samples} \
      --random_multibatch true \
      --shared_memory_option $7 \
      --shared_weight $9 \
      --numa_id $8 \
      --instance $5 \
      --server $6 \
      --mode ${11} \
      --include_accuracy true \
      --data_order NHWC \
      --net_conf $net_conf \
      ${netrun_settings}
