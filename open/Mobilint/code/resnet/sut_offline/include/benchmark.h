#ifndef MLPERF_V_1_0_INCLUDE_BENCHMARK_H_
#define MLPERF_V_1_0_INCLUDE_BENCHMARK_H_

#include <vector>
#include <memory>
#include "maccel.h"
#include "type.h"
#include "postprocessor.h"

extern std::vector<std::unique_ptr<mobilint::Accelerator>> mAccelerator;
extern maccel_type::Model SSDMobileNet, ResNet, SSDResNet;
extern uint8_t** loadedSample;
extern uint64_t* lookup;
extern vector<unique_ptr<PostprocessorManager>> mPostprocessor;

#endif