#!/bin/sh

make clean
#make MLP_INFER_ROOT=/workspace/mlperf-inference RT_ENGINE=/workspace/xcdl/Vitis-AI/rt-engine -j 
#make MLP_INFER_ROOT=/wrk/hdstaff/aaronn/git/mlperf-inference/ RT_ENGINE=/wrk/hdstaff/aaronn/git/xcdl/Vitis-AI/rt-engine -j 
make MLP_INFER_ROOT=/home/web/aaronn/03-21-dell-amd-mlperf.gaoyue/mlperf-inference/ RT_ENGINE=/home/web/aaronn/03-21-dell-amd-mlperf.gaoyue/rt-engine -j 
