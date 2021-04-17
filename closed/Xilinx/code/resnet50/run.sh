#!/bin/bash
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
export RTE_ACQUIRE_DEVICE_UNMANAGED=1
export XILINX_XRT=/opt/xilinx/xrt 
#export XLNX_VART_FIRMWARE=./v4e_8pe_100MHz_xilinx_vck5000-es1_gen3x16_2_202020_1_7e71129_Feb27Sat0922.xclbin
#export XLNX_VART_FIRMWARE=/home/web/aaronn/03-21-dell-amd-mlperf.gaoyue/rt-engine/v4e_8pe_350MHz_300MHz_xilinx_vck5000-es1_gen3x16_2_202020_1_24b1e3f_Mar02Tue1443.xclbin
#export XLNX_VART_FIRMWARE=/home/web/aaronn/03-21-dell-amd-mlperf.gaoyue/rt-engine/v4e_8pe_350MHz_300MHz_xilinx_vck5000-es1_gen3x16_2_202020_1_12dc3e8_Mar04Thu2025.xclbin
#export XLNX_VART_FIRMWARE=/home/web/aaronn/03-21-dell-amd-mlperf.gaoyue/rt-engine/v4e_8pe_350MHz_333MHz_xilinx_vck5000-es1_gen3x16_2_202020_1_fd6178d_Mar09Tue1420.xclbin
#export XLNX_VART_FIRMWARE=/home/web/aaronn/03-21-dell-amd-mlperf.gaoyue/rt-engine/v4e_8pe_375MHz_333MHz_xilinx_vck5000-es1_gen3x16_2_202020_1_54b855a_Mar10Wed2049.xclbin
export XLNX_VART_FIRMWARE=/home/web/aaronn/03-21-dell-amd-mlperf.gaoyue/rt-engine/v4e_8pe_375MHz_333MHz_xilinx_vck5000-es1_gen3x16_2_202020_1_67ffac7_Mar11Thu1637.xclbin
usage() {
  echo -e ""
  echo "Usage:"
  echo "------------------------------------------------"
  echo -e " ./run.sh " 
  echo -e "          --mode <mode-of-exe>" 
  echo -e "          --scenario <mlperf-screnario>"
  echo -e "          --dir <image-dir>"
  echo -e "          --nsamples <number-of-samples>"
  echo -e "          --qps <target qps>"
  echo -e ""
  echo "With Runner interface:"
  echo -e " ./run.sh --mode AccuracyOnly --scenario SingleStream"
  echo -e ""
}
# Default
MODE=PerformanceOnly
#DIRECTORY=/home/web/aaronn/imagenet/val/
DIRECTORY=/home/web/aaronn/val_dataset
SCENARIO=Server
SAMPLES=1024
TARGET_QPS=4000
MAX_ASYNC_QUERIES=200
MIN_TIME=60000

RT_ENGINE=${RT_ENGINE:=/home/web/aaronn/03-21-dell-amd-mlperf.gaoyue/rt-engine}
MLP_INFER_ROOT=${MLP_INFER_ROOT:=/home/web/aaronn/03-21-dell-amd-mlperf.gaoyue/mlperf-inference}
#DPU_DIR=${DPU_DIR:=model.dpuv4e/MLPerf_pruned_resnet_v1.5_remain0.74_finetune_compiled_DPUCVDX8H_ISA0.xmodel}
DPU_DIR=${DPU_DIR:=model.dpuv4e}
#DPU_DIR=${DPU_DIR:=../MLPerf_pruned_resnet_v1.5_remain0.74_finetune_compiled_DPUCVDX8H_ISA0.xmodel}
#DPU_DIR=${DPU_DIR:=./MLPerf_resnet50_v1.5_compiled_DPUCVDX8H_ISA0.xmodel}

# Parse Options
while true
do
  if [[ -z "$1" ]]; then break; fi
  case "$1" in
    -m  |--mode               ) MODE="$2"              ; shift 2 ;;
    -d  |--dir                ) DIRECTORY="$2"         ; shift 2 ;;
    -s  |--scenario           ) SCENARIO="$2"          ; shift 2 ;;
    -n  |--nsamples           ) SAMPLES="$2"           ; shift 2 ;;
    -r  |--qps                ) TARGET_QPS="$2"        ; shift 2 ;;
    -t  |--min_time           ) MIN_TIME="$2"          ; shift 2 ;;
    -a  |--max_async_queries  ) MAX_ASYNC_QUERIES="$2" ; shift 2 ;;
    -h  |--help               ) usage                  ; exit  1 ;;
     *) echo "Unknown argument : $1";
        exit 1 ;;
  esac
done

OPTIONS="--mode ${MODE} --scenario ${SCENARIO} --num_samples ${SAMPLES} --max_async_queries ${MAX_ASYNC_QUERIES} --qps ${TARGET_QPS} --min_time ${MIN_TIME}"

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CONDA_PREFIX}/lib:${RT_ENGINE}/build:/opt/xilinx/xrt/lib:${MLP_INFER_ROOT}/loadgen/build:/usr/local/lib64/:/usr/local/lib:${HOME}/.local/Ubuntu.18.04.x86_64.Release/lib

echo ${DPU_DIR}
echo  ${DIRECTORY} 
echo  ${OPTIONS}
./app.exe --dpudir ${DPU_DIR} --imgdir ${DIRECTORY} ${OPTIONS} 
#./app.exe --dpudir ${DPU_DIR}  ${OPTIONS} 

if [ "${MODE}" == "AccuracyOnly" ]
then
python3 ${MLP_INFER_ROOT}/vision/classification_and_detection/tools/accuracy-imagenet.py --imagenet-val-file=${DIRECTORY}/val_map.txt --mlperf-accuracy-file=mlperf_log_accuracy.json |& tee accuracy.txt
fi
