#!/bin/bash

SYSTEM_ID="dpuv4e"
COMPLIANCE_DIR="../mlperf-inference/compliance/nvidia"
USER_RUN_PARAMS="" # params passed to ./run.sh

# no pruning (1.33)
SERVER_TARGET_QPS="5920"
OFFLINE_TARGET_QPS="6400"

# no pruning (1.5)
#SERVER_TARGET_QPS="6470"
#OFFLINE_TARGET_QPS="7020"

while true
do
  if [[ -z "$1" ]]; then break; fi
  case "$1" in
    -p  |--params             ) USER_RUN_PARAMS="$2"    ; shift 2 ;;
     *) echo "Unknown argument : $1";
        exit 1 ;;
  esac
done

for SCENARIO in Server Offline
do
  RESULT_DIR="results/${SYSTEM_ID}/resnet50/${SCENARIO}"
  AUDIT_DIR="compliance/${SYSTEM_ID}/resnet50/${SCENARIO}"
  OUTPUT_DIR="compliance_output/${SYSTEM_ID}/resnet50/${SCENARIO}"

  if [ ${SCENARIO} = "Offline" ]
  then
    NUM_ITER=1
    TARGET_QPS=${OFFLINE_TARGET_QPS}
    RUN_PARAMS="-n 24576 ${USER_RUN_PARAMS}"
  else
    NUM_ITER=5
    TARGET_QPS=${SERVER_TARGET_QPS}
    RUN_PARAMS="-n 1024 ${USER_RUN_PARAMS}"
  fi
  
  # run normally (without audit.config)
  mkdir -p ${RESULT_DIR}/accuracy
  echo -e "\nRunning ${SCENARIO} accuracy"
  ./run.sh -s ${SCENARIO} -m AccuracyOnly ${RUN_PARAMS} -n 50000 -r ${TARGET_QPS} -t 600000
  cp mlperf_log*.{txt,json} accuracy.txt ${RESULT_DIR}/accuracy
  for (( ITER=1; ITER<=$NUM_ITER; ITER++ ))
  do
    mkdir -p ${RESULT_DIR}/performance/run_${ITER}
    echo -e "\nRunning ${SCENARIO} performance #${ITER}"
    ./run.sh -s ${SCENARIO} -m PerformanceOnly ${RUN_PARAMS} -r ${TARGET_QPS} -t 600000
    cp mlperf_log*.{txt,json} ${RESULT_DIR}/performance/run_${ITER}
  done
  
  # TEST01
  echo -e "\nRunning ${SCENARIO} TEST01 compliance"
  mkdir -p ${AUDIT_DIR}/TEST01/{performance,accuracy}
  cp ${COMPLIANCE_DIR}/TEST01/resnet50/audit.config .
  ./run.sh -s ${SCENARIO} -m PerformanceOnly ${RUN_PARAMS} -r ${TARGET_QPS} -t 600000
  cp mlperf_log*.{txt,json} accuracy.txt ${AUDIT_DIR}/TEST01/accuracy
  cp mlperf_log*.{txt,json} ${AUDIT_DIR}/TEST01/performance
  python3 ${COMPLIANCE_DIR}/TEST01/run_verification.py -r ${RESULT_DIR} -o ${AUDIT_DIR} --dtype float32 -c ${AUDIT_DIR}/TEST01/performance
  rm audit.config
  
  # TEST04-A and TEST04-B
  echo -e "\nRunning ${SCENARIO} TEST04 compliance"
  for item in TEST04-A TEST04-B 
  do
    mkdir -p ${AUDIT_DIR}/${item}/{performance,accuracy}
    cp ${COMPLIANCE_DIR}/${item}/audit.config .
    ./run.sh -s ${SCENARIO} -m PerformanceOnly ${RUN_PARAMS} -r ${TARGET_QPS} -t 600000
    cp mlperf_log*.{txt,json} accuracy.txt ${AUDIT_DIR}/${item}/accuracy
    cp mlperf_log*.{txt,json} ${AUDIT_DIR}/${item}/performance
    rm audit.config
  done
  python3 ${COMPLIANCE_DIR}/TEST04-A/run_verification.py -a ${AUDIT_DIR}/TEST04-A/performance -b ${AUDIT_DIR}/TEST04-B/performance -o ${AUDIT_DIR} 
  
  # TEST05
  echo -e "\nRunning ${SCENARIO} TEST05 compliance"
  mkdir -p ${AUDIT_DIR}/TEST05/{performance,accuracy}
  cp ${COMPLIANCE_DIR}/TEST05/audit.config .
  ./run.sh -s ${SCENARIO} -m PerformanceOnly ${RUN_PARAMS} -r ${TARGET_QPS} -t 600000
  cp mlperf_log*.{txt,json} accuracy.txt ${AUDIT_DIR}/TEST05/accuracy
  cp mlperf_log*.{txt,json} ${AUDIT_DIR}/TEST05/performance
  python3 ${COMPLIANCE_DIR}/TEST05/run_verification.py -r ${RESULT_DIR} -o ${AUDIT_DIR} -c ${AUDIT_DIR}/TEST05/performance
  rm audit.config
done
