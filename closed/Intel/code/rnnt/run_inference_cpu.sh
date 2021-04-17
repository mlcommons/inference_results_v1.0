# tcmalloc:
#export LD_PRELOAD=$TCMALLOC_DIR/libtcmalloc.so

# jemalloc
export LD_PRELOAD=$TCMALLOC_DIR/libjemalloc.so
#export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
export MALLOC_CONF="background_thread:true,dirty_decay_ms:8000,muzzy_decay_ms:8000"

sockets=`lscpu | grep Socket | awk '{print $2}'`
cores=`lscpu | grep Core.*per\ socket: | awk '{print $4}'`

root_dir=`pwd`
work_dir=$root_dir/mlperf-rnnt-librispeech
local_data_dir=$work_dir/local_data

scenario=Offline
if [[ "$1" == "--server" ]]; then
    scenario=Server
    shift
fi

debug=""
if [[ "$1" == "--debug" ]]; then
    debug="--debug"
    shift
fi

accuracy=""
if [[ "$1" == "--accuracy" ]]; then
	    debug="--accuracy"
	        shift
fi

if [[ "$scenario" == "Server" ]]; then
  batch_size=$BATCH_SIZE
  cores_per_instance=$CPUS_PER_INSTANCE
  cores_for_loadgen=0
  #warmup="--warmup True"
else
  batch_size=$BATCH_SIZE
  cores_per_instance=$CPUS_PER_INSTANCE
  cores_for_loadgen=0
fi

instances_per_socket=`expr $cores \/ $cores_per_instance`
num_instances=`expr $sockets \* $instances_per_socket`
#num_instances=`expr 1 \* $instances_per_socket`
#num_instances=224

backend=pytorch

log_dir=${work_dir}/${scenario}_${backend}
if [ ! -z ${accuracy} ]; then
    log_dir+=_accuracy
fi
log_dir+=rerun

export DNNL_PRIMITIVE_CACHE_CAPACITY=10485760

python run.py --dataset_dir $local_data_dir \
    --manifest $local_data_dir/dev-clean-wav.json \
    --pytorch_config_toml pytorch/configs/rnnt.toml \
    --pytorch_checkpoint $work_dir/rnnt.pt \
    --scenario ${scenario} \
    --backend ${backend} \
    --log_dir output \
    --ipex True \
    --bf16 True \
    --offline_batch_size ${batch_size} \
    --cores_for_loadgen $cores_for_loadgen \
    --cores_per_instance $cores_per_instance \
    ${accuracy} \
    ${warmup} \
    ${debug}
