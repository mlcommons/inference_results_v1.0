/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <iostream>
#include <sstream>
#include <cstring>
#include <vector>
#include <numeric>
#include <fstream>
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <dlfcn.h>

#include <cudnn.h>
#include <nvrtc.h>

#include "dlrmBottomMLPPlugin.h"
#include "dlrmHelper.h"

using namespace nvinfer1;

const int DLRMBottomMLPPlugin::mWarpsInThreadblock = 4;

const char * DLRMBottomMLPPlugin::mBottomMLPFusionKernelCode =
"#include <cuda_fp16.h> \n\
#include <mma.h> \n\
#if __CUDA_ARCH__ >= 800 \n\
#include <cuda_pipeline.h> \n\
#endif \n\
\n\
using namespace nvcuda; \n\
\n\
union int8pack \n\
{ \n\
    int v; \n\
    char t[4]; \n\
}; \n\
\n\
__forceinline__ __device__ float slowIntToFloat(int v) \n\
{ \n\
    return __int2float_rn(v); \n\
} \n\
\n\
__forceinline__ __device__ float fastIntToFloat(int v) \n\
{ \n\
    return __int_as_float(v) - 12582912.0F; \n\
} \n\
\n\
__forceinline__ __device__ int getAccInit(int k) \n\
{ \n\
    if (k <= 256) \n\
        return 0x4B400000; \n\
    else \n\
        return 0; \n\
} \n\
\n\
__forceinline__ __device__ float intToFloat(int v, int k) \n\
{ \n\
    if (k <= 256) \n\
        return fastIntToFloat(v); \n\
    else \n\
        return slowIntToFloat(v); \n\
} \n\
\n\
__forceinline__ __device__ char slowFloatReLUToInt8(float v) \n\
{ \n\
    int8pack res; \n\
    res.v = __float2int_rn(max(min(v, 127.0F), 0.0F)); \n\
    return res.t[0]; \n\
} \n\
\n\
__forceinline__ __device__ char fastFloatReLUToInt8(float v) \n\
{ \n\
    int8pack res; \n\
    res.v = __float_as_int(max(min(v, 127.0F), 0.0F) + 8388608.0F); \n\
    return res.t[0]; \n\
} \n\
\n\
__forceinline__ __device__ int floatReLUToInt(float v) \n\
{ \n\
    return __float2int_rn(max(v, 0.0F)); \n\
} \n\
\n\
__forceinline__ __device__ int packFloatReLUToInt8(float x, float y, float z, float w) \n\
{ \n\
#if __CUDA_ARCH__ >= 860 \n\
    int res; \n\
    int xi = floatReLUToInt(x); \n\
    int yi = floatReLUToInt(y); \n\
    int zi = floatReLUToInt(z); \n\
    int wi = floatReLUToInt(w); \n\
    asm(\" { .reg .u32 t1; cvt.pack.sat.s8.s32.b32 t1, %1, %2, 0; cvt.pack.sat.s8.s32.b32 %0, %3, %4, t1; } \" : \"=r\"(res) : \"r\"(wi), \"r\"(zi), \"r\"(yi), \"r\"(xi)); \n\
    return res; \n\
#else \n\
    int8pack res; \n\
    res.t[0] = fastFloatReLUToInt8(x); \n\
    res.t[1] = fastFloatReLUToInt8(y); \n\
    res.t[2] = fastFloatReLUToInt8(z); \n\
    res.t[3] = fastFloatReLUToInt8(w); \n\
    return res.v; \n\
#endif \n\
} \n\
\n\
extern \"C\" \n\
__launch_bounds__(THREADBLOCK_SIZE,256/THREADBLOCK_SIZE) \n\
__global__ void fusedbottommlpkernel( \n\
    const char * __restrict srcActivations, \n\
    const char * __restrict weights1, \n\
    const float * __restrict biases1, \n\
    float scale1, \n\
    const char * __restrict weights2, \n\
    const float * __restrict biases2, \n\
    float scale2, \n\
    const char * __restrict weights3, \n\
    const float * __restrict biases3, \n\
    float scale3, \n\
    char * __restrict dstActivations, \n\
    int batchSize) \n\
{ \n\
#if __CUDA_ARCH__ >= 750 \n\
    int baseSampleId = (blockIdx.x * THREADBLOCK_SIZE  + threadIdx.x) / 32 * 16; \n\
    if (baseSampleId >= batchSize) \n\
        return; \n\
    int laneId = threadIdx.x & 31; \n\
\n\
    int2 input[ELEMENT_COUNT_0_PADDED / 16]; \n\
    { \n\
        const int * baseSrcActivations = (const int *)(srcActivations + baseSampleId * ELEMENT_COUNT_0); \n\
        int offsetSampleId = laneId / 4; \n\
        int sampleId = baseSampleId + offsetSampleId; \n\
        if (ELEMENT_COUNT_0 <= 16) \n\
        { \n\
            // Optimization for the smallest case \n\
            input[0].x = ((sampleId < batchSize) && ((ELEMENT_COUNT_0 == ELEMENT_COUNT_0_PADDED) || (laneId < ELEMENT_COUNT_0 / 4))) ? baseSrcActivations[laneId] : 0; \n\
            sampleId += 8; \n\
            input[0].y = ((sampleId < batchSize) && ((ELEMENT_COUNT_0 == ELEMENT_COUNT_0_PADDED) || (laneId < ELEMENT_COUNT_0 / 4))) ? baseSrcActivations[laneId + 32] : 0; \n\
        } \n\
        else \n\
        { \n\
            int offsetElemId = laneId % 4; \n\
#pragma unroll \n\
            for(int fragmentId = 0; fragmentId < ELEMENT_COUNT_0 / 16; ++fragmentId) \n\
                input[fragmentId].x = (sampleId < batchSize) ? baseSrcActivations[offsetSampleId * (ELEMENT_COUNT_0 / 4) + offsetElemId + fragmentId * 4] : 0; \n\
            if (ELEMENT_COUNT_0 != ELEMENT_COUNT_0_PADDED) \n\
                input[ELEMENT_COUNT_0 / 16].x = ((sampleId < batchSize) && ((offsetElemId + (ELEMENT_COUNT_0 / 16) * 4) < ELEMENT_COUNT_0 / 4)) ? \n\
                    baseSrcActivations[offsetSampleId * (ELEMENT_COUNT_0 / 4) + offsetElemId + (ELEMENT_COUNT_0 / 16) * 4] : 0; \n\
            sampleId += 8; \n\
#pragma unroll \n\
            for(int fragmentId = 0; fragmentId < ELEMENT_COUNT_0 / 16; ++fragmentId) \n\
                input[fragmentId].y = (sampleId < batchSize) ? baseSrcActivations[offsetSampleId * (ELEMENT_COUNT_0 / 4) + ELEMENT_COUNT_0 * 2 + offsetElemId + fragmentId * 4] : 0; \n\
            if (ELEMENT_COUNT_0 != ELEMENT_COUNT_0_PADDED) \n\
                input[ELEMENT_COUNT_0 / 16].y = ((sampleId < batchSize) && ((offsetElemId + (ELEMENT_COUNT_0 / 16) * 4) < ELEMENT_COUNT_0 / 4)) ? \n\
                    baseSrcActivations[offsetSampleId * (ELEMENT_COUNT_0 / 4) + ELEMENT_COUNT_0 * 2 + offsetElemId + (ELEMENT_COUNT_0 / 16) * 4] : 0; \n\
        } \n\
    } \n\
\n\
    int4 inputLayer2[ELEMENT_COUNT_1 / 32]; \n\
#pragma unroll \n\
    for(int fragmentId = 0; fragmentId < ELEMENT_COUNT_1 / 8; fragmentId += 4) \n\
    { \n\
        int4 acc[4]; \n\
        acc[0] = make_int4(getAccInit(ELEMENT_COUNT_0), getAccInit(ELEMENT_COUNT_0), getAccInit(ELEMENT_COUNT_0), getAccInit(ELEMENT_COUNT_0)); \n\
        acc[1] = make_int4(getAccInit(ELEMENT_COUNT_0), getAccInit(ELEMENT_COUNT_0), getAccInit(ELEMENT_COUNT_0), getAccInit(ELEMENT_COUNT_0)); \n\
        acc[2] = make_int4(getAccInit(ELEMENT_COUNT_0), getAccInit(ELEMENT_COUNT_0), getAccInit(ELEMENT_COUNT_0), getAccInit(ELEMENT_COUNT_0)); \n\
        acc[3] = make_int4(getAccInit(ELEMENT_COUNT_0), getAccInit(ELEMENT_COUNT_0), getAccInit(ELEMENT_COUNT_0), getAccInit(ELEMENT_COUNT_0)); \n\
        const int4 * baseWeights = (const int4 *)(weights1 + fragmentId * (ELEMENT_COUNT_0_PADDED * 8)); \n\
#pragma unroll \n\
        for(int k = 0; k < ELEMENT_COUNT_0_PADDED / 16; ++k) \n\
        { \n\
            int4 weights = baseWeights[laneId + k * 32]; \n\
#if __CUDA_ARCH__ >= 800 && !defined FORCE_TURING_IMMA \n\
            asm(\"mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};\" : \"+r\"(acc[0].x), \"+r\"(acc[0].y), \"+r\"(acc[0].z), \"+r\"(acc[0].w) : \"r\"(input[k].x), \"r\"(input[k].y), \"r\"(weights.x)); \n\
            asm(\"mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};\" : \"+r\"(acc[1].x), \"+r\"(acc[1].y), \"+r\"(acc[1].z), \"+r\"(acc[1].w) : \"r\"(input[k].x), \"r\"(input[k].y), \"r\"(weights.z)); \n\
            asm(\"mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};\" : \"+r\"(acc[2].x), \"+r\"(acc[2].y), \"+r\"(acc[2].z), \"+r\"(acc[2].w) : \"r\"(input[k].x), \"r\"(input[k].y), \"r\"(weights.y)); \n\
            asm(\"mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};\" : \"+r\"(acc[3].x), \"+r\"(acc[3].y), \"+r\"(acc[3].z), \"+r\"(acc[3].w) : \"r\"(input[k].x), \"r\"(input[k].y), \"r\"(weights.w)); \n\
#else \n\
            asm(\"mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};\" : \"+r\"(acc[0].x), \"+r\"(acc[0].y) : \"r\"(input[k].x), \"r\"(weights.x)); \n\
            asm(\"mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};\" : \"+r\"(acc[0].z), \"+r\"(acc[0].w) : \"r\"(input[k].y), \"r\"(weights.x)); \n\
            asm(\"mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};\" : \"+r\"(acc[1].x), \"+r\"(acc[1].y) : \"r\"(input[k].x), \"r\"(weights.z)); \n\
            asm(\"mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};\" : \"+r\"(acc[1].z), \"+r\"(acc[1].w) : \"r\"(input[k].y), \"r\"(weights.z)); \n\
            asm(\"mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};\" : \"+r\"(acc[2].x), \"+r\"(acc[2].y) : \"r\"(input[k].x), \"r\"(weights.y)); \n\
            asm(\"mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};\" : \"+r\"(acc[2].z), \"+r\"(acc[2].w) : \"r\"(input[k].y), \"r\"(weights.y)); \n\
            asm(\"mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};\" : \"+r\"(acc[3].x), \"+r\"(acc[3].y) : \"r\"(input[k].x), \"r\"(weights.w)); \n\
            asm(\"mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};\" : \"+r\"(acc[3].z), \"+r\"(acc[3].w) : \"r\"(input[k].y), \"r\"(weights.w)); \n\
#endif \n\
        } \n\
        const float4 * baseBiases = (const float4 *)biases1 + fragmentId * 2; \n\
        float4 biases1 = baseBiases[laneId & 3]; \n\
        float4 biases2 = baseBiases[(laneId & 3) + 4]; \n\
        inputLayer2[fragmentId / 4].x = packFloatReLUToInt8(intToFloat(acc[0].x, ELEMENT_COUNT_0) * scale1 + biases1.x, \n\
            intToFloat(acc[0].y, ELEMENT_COUNT_0) * scale1 + biases1.y, \n\
            intToFloat(acc[1].x, ELEMENT_COUNT_0) * scale1 + biases1.z, \n\
            intToFloat(acc[1].y, ELEMENT_COUNT_0) * scale1 + biases1.w); \n\
        inputLayer2[fragmentId / 4].y = packFloatReLUToInt8(intToFloat(acc[0].z, ELEMENT_COUNT_0) * scale1 + biases1.x, \n\
            intToFloat(acc[0].w, ELEMENT_COUNT_0) * scale1 + biases1.y, \n\
            intToFloat(acc[1].z, ELEMENT_COUNT_0) * scale1 + biases1.z, \n\
            intToFloat(acc[1].w, ELEMENT_COUNT_0) * scale1 + biases1.w); \n\
        inputLayer2[fragmentId / 4].z = packFloatReLUToInt8(intToFloat(acc[2].x, ELEMENT_COUNT_0) * scale1 + biases2.x, \n\
            intToFloat(acc[2].y, ELEMENT_COUNT_0) * scale1 + biases2.y, \n\
            intToFloat(acc[3].x, ELEMENT_COUNT_0) * scale1 + biases2.z, \n\
            intToFloat(acc[3].y, ELEMENT_COUNT_0) * scale1 + biases2.w); \n\
        inputLayer2[fragmentId / 4].w = packFloatReLUToInt8(intToFloat(acc[2].z, ELEMENT_COUNT_0) * scale1 + biases2.x, \n\
            intToFloat(acc[2].w, ELEMENT_COUNT_0) * scale1 + biases2.y, \n\
            intToFloat(acc[3].z, ELEMENT_COUNT_0) * scale1 + biases2.z, \n\
            intToFloat(acc[3].w, ELEMENT_COUNT_0) * scale1 + biases2.w); \n\
    } \n\
\n\
    int4 inputLayer3[ELEMENT_COUNT_2 / 32]; \n\
#pragma unroll \n\
    for(int fragmentId = 0; fragmentId < ELEMENT_COUNT_2 / 8; fragmentId += 4) \n\
    { \n\
        int4 acc[4]; \n\
        acc[0] = make_int4(getAccInit(ELEMENT_COUNT_1), getAccInit(ELEMENT_COUNT_1), getAccInit(ELEMENT_COUNT_1), getAccInit(ELEMENT_COUNT_1)); \n\
        acc[1] = make_int4(getAccInit(ELEMENT_COUNT_1), getAccInit(ELEMENT_COUNT_1), getAccInit(ELEMENT_COUNT_1), getAccInit(ELEMENT_COUNT_1)); \n\
        acc[2] = make_int4(getAccInit(ELEMENT_COUNT_1), getAccInit(ELEMENT_COUNT_1), getAccInit(ELEMENT_COUNT_1), getAccInit(ELEMENT_COUNT_1)); \n\
        acc[3] = make_int4(getAccInit(ELEMENT_COUNT_1), getAccInit(ELEMENT_COUNT_1), getAccInit(ELEMENT_COUNT_1), getAccInit(ELEMENT_COUNT_1)); \n\
        const int4 * baseWeights = (const int4 *)(weights2 + fragmentId * (ELEMENT_COUNT_1 * 8)); \n\
#pragma unroll \n\
        for(int k = 0; k < ELEMENT_COUNT_1 / 32; ++k) \n\
        { \n\
            int4 weights1 = baseWeights[laneId + k * 32]; \n\
            int4 weights2 = baseWeights[laneId + ELEMENT_COUNT_1 + k * 32]; \n\
#if __CUDA_ARCH__ >= 800 && !defined FORCE_TURING_IMMA \n\
            asm(\"mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\" : \"+r\"(acc[0].x), \"+r\"(acc[0].y), \"+r\"(acc[0].z), \"+r\"(acc[0].w) : \"r\"(inputLayer2[k].x), \"r\"(inputLayer2[k].y), \"r\"(inputLayer2[k].z), \"r\"(inputLayer2[k].w), \"r\"(weights1.x), \"r\"(weights1.y)); \n\
            asm(\"mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\" : \"+r\"(acc[1].x), \"+r\"(acc[1].y), \"+r\"(acc[1].z), \"+r\"(acc[1].w) : \"r\"(inputLayer2[k].x), \"r\"(inputLayer2[k].y), \"r\"(inputLayer2[k].z), \"r\"(inputLayer2[k].w), \"r\"(weights1.z), \"r\"(weights1.w)); \n\
            asm(\"mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\" : \"+r\"(acc[2].x), \"+r\"(acc[2].y), \"+r\"(acc[2].z), \"+r\"(acc[2].w) : \"r\"(inputLayer2[k].x), \"r\"(inputLayer2[k].y), \"r\"(inputLayer2[k].z), \"r\"(inputLayer2[k].w), \"r\"(weights2.x), \"r\"(weights2.y)); \n\
            asm(\"mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\" : \"+r\"(acc[3].x), \"+r\"(acc[3].y), \"+r\"(acc[3].z), \"+r\"(acc[3].w) : \"r\"(inputLayer2[k].x), \"r\"(inputLayer2[k].y), \"r\"(inputLayer2[k].z), \"r\"(inputLayer2[k].w), \"r\"(weights2.z), \"r\"(weights2.w)); \n\
#else \n\
            asm(\"mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};\" : \"+r\"(acc[0].x), \"+r\"(acc[0].y) : \"r\"(inputLayer2[k].x), \"r\"(weights1.x)); \n\
            asm(\"mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};\" : \"+r\"(acc[0].z), \"+r\"(acc[0].w) : \"r\"(inputLayer2[k].y), \"r\"(weights1.x)); \n\
            asm(\"mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};\" : \"+r\"(acc[1].x), \"+r\"(acc[1].y) : \"r\"(inputLayer2[k].x), \"r\"(weights1.z)); \n\
            asm(\"mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};\" : \"+r\"(acc[1].z), \"+r\"(acc[1].w) : \"r\"(inputLayer2[k].y), \"r\"(weights1.z)); \n\
            asm(\"mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};\" : \"+r\"(acc[0].x), \"+r\"(acc[0].y) : \"r\"(inputLayer2[k].z), \"r\"(weights1.y)); \n\
            asm(\"mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};\" : \"+r\"(acc[0].z), \"+r\"(acc[0].w) : \"r\"(inputLayer2[k].w), \"r\"(weights1.y)); \n\
            asm(\"mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};\" : \"+r\"(acc[1].x), \"+r\"(acc[1].y) : \"r\"(inputLayer2[k].z), \"r\"(weights1.w)); \n\
            asm(\"mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};\" : \"+r\"(acc[1].z), \"+r\"(acc[1].w) : \"r\"(inputLayer2[k].w), \"r\"(weights1.w)); \n\
            asm(\"mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};\" : \"+r\"(acc[2].x), \"+r\"(acc[2].y) : \"r\"(inputLayer2[k].x), \"r\"(weights2.x)); \n\
            asm(\"mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};\" : \"+r\"(acc[2].z), \"+r\"(acc[2].w) : \"r\"(inputLayer2[k].y), \"r\"(weights2.x)); \n\
            asm(\"mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};\" : \"+r\"(acc[3].x), \"+r\"(acc[3].y) : \"r\"(inputLayer2[k].x), \"r\"(weights2.z)); \n\
            asm(\"mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};\" : \"+r\"(acc[3].z), \"+r\"(acc[3].w) : \"r\"(inputLayer2[k].y), \"r\"(weights2.z)); \n\
            asm(\"mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};\" : \"+r\"(acc[2].x), \"+r\"(acc[2].y) : \"r\"(inputLayer2[k].z), \"r\"(weights2.y)); \n\
            asm(\"mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};\" : \"+r\"(acc[2].z), \"+r\"(acc[2].w) : \"r\"(inputLayer2[k].w), \"r\"(weights2.y)); \n\
            asm(\"mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};\" : \"+r\"(acc[3].x), \"+r\"(acc[3].y) : \"r\"(inputLayer2[k].z), \"r\"(weights2.w)); \n\
            asm(\"mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};\" : \"+r\"(acc[3].z), \"+r\"(acc[3].w) : \"r\"(inputLayer2[k].w), \"r\"(weights2.w)); \n\
#endif \n\
        } \n\
        const float4 * baseBiases = (const float4 *)biases2 + fragmentId * 2; \n\
        float4 biases1 = baseBiases[laneId & 3]; \n\
        float4 biases2 = baseBiases[(laneId & 3) + 4]; \n\
        // Use slow int=>float conversion as the accumulator might be too big for the fast one \n\
        inputLayer3[fragmentId / 4].x = packFloatReLUToInt8(intToFloat(acc[0].x, ELEMENT_COUNT_1) * scale2 + biases1.x, \n\
            intToFloat(acc[0].y, ELEMENT_COUNT_1) * scale2 + biases1.y, \n\
            intToFloat(acc[1].x, ELEMENT_COUNT_1) * scale2 + biases1.z, \n\
            intToFloat(acc[1].y, ELEMENT_COUNT_1) * scale2 + biases1.w); \n\
        inputLayer3[fragmentId / 4].y = packFloatReLUToInt8(intToFloat(acc[0].z, ELEMENT_COUNT_1) * scale2 + biases1.x, \n\
            intToFloat(acc[0].w, ELEMENT_COUNT_1) * scale2 + biases1.y, \n\
            intToFloat(acc[1].z, ELEMENT_COUNT_1) * scale2 + biases1.z, \n\
            intToFloat(acc[1].w, ELEMENT_COUNT_1) * scale2 + biases1.w); \n\
        inputLayer3[fragmentId / 4].z = packFloatReLUToInt8(intToFloat(acc[2].x, ELEMENT_COUNT_1) * scale2 + biases2.x, \n\
            intToFloat(acc[2].y, ELEMENT_COUNT_1) * scale2 + biases2.y, \n\
            intToFloat(acc[3].x, ELEMENT_COUNT_1) * scale2 + biases2.z, \n\
            intToFloat(acc[3].y, ELEMENT_COUNT_1) * scale2 + biases2.w); \n\
        inputLayer3[fragmentId / 4].w = packFloatReLUToInt8(intToFloat(acc[2].z, ELEMENT_COUNT_1) * scale2 + biases2.x, \n\
            intToFloat(acc[2].w, ELEMENT_COUNT_1) * scale2 + biases2.y, \n\
            intToFloat(acc[3].z, ELEMENT_COUNT_1) * scale2 + biases2.z, \n\
            intToFloat(acc[3].w, ELEMENT_COUNT_1) * scale2 + biases2.w); \n\
    } \n\
\n\
#pragma unroll 4 \n\
    for(int fragmentId = 0; fragmentId < ELEMENT_COUNT_3 / 8; fragmentId += 2) \n\
    { \n\
        int4 acc[2]; \n\
        acc[0] = make_int4(getAccInit(ELEMENT_COUNT_2), getAccInit(ELEMENT_COUNT_2), getAccInit(ELEMENT_COUNT_2), getAccInit(ELEMENT_COUNT_2)); \n\
        acc[1] = make_int4(getAccInit(ELEMENT_COUNT_2), getAccInit(ELEMENT_COUNT_2), getAccInit(ELEMENT_COUNT_2), getAccInit(ELEMENT_COUNT_2)); \n\
        const int4 * baseWeights = (const int4 *)(weights3 + fragmentId * (ELEMENT_COUNT_2 * 8)); \n\
#pragma unroll \n\
        for(int k = 0; k < ELEMENT_COUNT_2 / 32; ++k) \n\
        { \n\
            int4 weights = baseWeights[laneId + k * 32]; \n\
#if __CUDA_ARCH__ >= 800 && !defined FORCE_TURING_IMMA \n\
            asm(\"mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\" : \"+r\"(acc[0].x), \"+r\"(acc[0].y), \"+r\"(acc[0].z), \"+r\"(acc[0].w) : \"r\"(inputLayer3[k].x), \"r\"(inputLayer3[k].y), \"r\"(inputLayer3[k].z), \"r\"(inputLayer3[k].w), \"r\"(weights.x), \"r\"(weights.y)); \n\
            asm(\"mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\" : \"+r\"(acc[1].x), \"+r\"(acc[1].y), \"+r\"(acc[1].z), \"+r\"(acc[1].w) : \"r\"(inputLayer3[k].x), \"r\"(inputLayer3[k].y), \"r\"(inputLayer3[k].z), \"r\"(inputLayer3[k].w), \"r\"(weights.z), \"r\"(weights.w)); \n\
#else \n\
            asm(\"mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};\" : \"+r\"(acc[0].x), \"+r\"(acc[0].y) : \"r\"(inputLayer3[k].x), \"r\"(weights.x)); \n\
            asm(\"mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};\" : \"+r\"(acc[0].z), \"+r\"(acc[0].w) : \"r\"(inputLayer3[k].y), \"r\"(weights.x)); \n\
            asm(\"mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};\" : \"+r\"(acc[1].x), \"+r\"(acc[1].y) : \"r\"(inputLayer3[k].x), \"r\"(weights.z)); \n\
            asm(\"mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};\" : \"+r\"(acc[1].z), \"+r\"(acc[1].w) : \"r\"(inputLayer3[k].y), \"r\"(weights.z)); \n\
            asm(\"mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};\" : \"+r\"(acc[0].x), \"+r\"(acc[0].y) : \"r\"(inputLayer3[k].z), \"r\"(weights.y)); \n\
            asm(\"mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};\" : \"+r\"(acc[0].z), \"+r\"(acc[0].w) : \"r\"(inputLayer3[k].w), \"r\"(weights.y)); \n\
            asm(\"mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};\" : \"+r\"(acc[1].x), \"+r\"(acc[1].y) : \"r\"(inputLayer3[k].z), \"r\"(weights.w)); \n\
            asm(\"mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};\" : \"+r\"(acc[1].z), \"+r\"(acc[1].w) : \"r\"(inputLayer3[k].w), \"r\"(weights.w)); \n\
#endif \n\
        } \n\
\n\
        const float4 * baseBiases = (const float4 *)biases3 + fragmentId * 2; \n\
        float4 biases = baseBiases[laneId & 3]; \n\
        int outputMerged[2]; \n\
        outputMerged[0] = packFloatReLUToInt8(intToFloat(acc[0].x, ELEMENT_COUNT_2) * scale3 + biases.x, \n\
            intToFloat(acc[0].y, ELEMENT_COUNT_2) * scale3 + biases.y, \n\
            intToFloat(acc[1].x, ELEMENT_COUNT_2) * scale3 + biases.z, \n\
            intToFloat(acc[1].y, ELEMENT_COUNT_2) * scale3 + biases.w); \n\
        outputMerged[1] = packFloatReLUToInt8(intToFloat(acc[0].z, ELEMENT_COUNT_2) * scale3 + biases.x, \n\
            intToFloat(acc[0].w, ELEMENT_COUNT_2) * scale3 + biases.y, \n\
            intToFloat(acc[1].z, ELEMENT_COUNT_2) * scale3 + biases.z, \n\
            intToFloat(acc[1].w, ELEMENT_COUNT_2) * scale3 + biases.w); \n\
\n\
        int localRowId = laneId / 4; \n\
        int localColId = laneId & 3; \n\
        int * baseDstActivations = ((int *)dstActivations) + (baseSampleId * 32 + fragmentId * 2 + localColId); \n\
        if (baseSampleId + localRowId < batchSize) \n\
            baseDstActivations[localRowId * 32] = outputMerged[0]; \n\
        localRowId += 8; \n\
        if (baseSampleId + localRowId < batchSize) \n\
            baseDstActivations[localRowId * 32] = outputMerged[1]; \n\
    } \n\
#else \n\
    assert(0); \n\
#endif \n\
}";

namespace
{
const char* DLRM_BOTTOM_MLP_PLUGIN_VERSION{"1"};
const char* DLRM_BOTTOM_MLP_PLUGIN_NAME{"DLRM_BOTTOM_MLP_TRT"};
}

PluginFieldCollection DLRMBottomMLPPluginCreator::mFC{};
std::vector<PluginField> DLRMBottomMLPPluginCreator::mPluginAttributes;
REGISTER_TENSORRT_PLUGIN(DLRMBottomMLPPluginCreator);

DLRMBottomMLPPlugin::DLRMBottomMLPPlugin(
    int inputChannels,
    const std::vector<std::vector<float>>& weights,
    const std::vector<std::vector<float>>& biases,
    const std::vector<float>& dynamicRanges)
    : mInitialized(false)
    , mInputChannels((inputChannels + 3) / 4 * 4)
    , mBiases(biases)
{
    ASSERT(biases.size() == 3);
    ASSERT((biases[2].size() % 32) == 0);

    mActivationScales.resize(4);
    mActivationScales[1] = dynamicRanges[0] / 127.0F;
    mActivationScales[2] = dynamicRanges[1] / 127.0F;

    mWeightScales.resize(weights.size());
    mWeights.resize(weights.size());
    for(int i = 0; i < static_cast<int>(weights.size()); ++i)
    {
        float maxAbsVal = std::accumulate(weights[i].begin(), weights[i].end(), 0.0F, [] (float x, float y) { return std::max(fabsf(x), fabsf(y)); });
        float mult = 127.0F / maxAbsVal;
        mWeightScales[i] = 1.0F / mult;
        int rows = biases[i].size();
        int originalColumns = (i == 0) ? inputChannels : biases[i - 1].size();
        int paddedColumns = (i == 0) ? mInputChannels : biases[i - 1].size();
        mWeights[i].resize(rows * paddedColumns);
        for(int row = 0; row < rows; ++row)
            std::transform(weights[i].begin() + row * originalColumns, weights[i].begin() + (row + 1) * originalColumns,
                mWeights[i].begin() + row * paddedColumns,
                [=] (float x) {return static_cast<int8_t>(roundf(std::max(std::min(x * mult, 127.0F), -127.0F))); });
    }
}

DLRMBottomMLPPlugin::DLRMBottomMLPPlugin(const void* data, size_t length)
    : mInitialized(false)
{
    const char* d = reinterpret_cast<const char *>(data);
    const char* a = d;

    int temp;
    mBiases.resize(3);
    mInputChannels = read<int>(d);
    temp = read<int>(d);
    mBiases[0].resize(temp);
    temp = read<int>(d);
    mBiases[1].resize(temp);
    temp = read<int>(d);
    mBiases[2].resize(temp);
    mActivationScales.resize(4);
    read(d, &mActivationScales.front(), mActivationScales.size());
    mWeightScales.resize(3);
    read(d, &mWeightScales.front(), mWeightScales.size());
    read(d, &mBiases[0].front(), mBiases[0].size());
    read(d, &mBiases[1].front(), mBiases[1].size());
    read(d, &mBiases[2].front(), mBiases[2].size());
    mWeights.resize(3);
    mWeights[0].resize(mInputChannels * mBiases[0].size());
    read(d, &mWeights[0].front(), mWeights[0].size());
    mWeights[1].resize(mBiases[0].size() * mBiases[1].size());
    read(d, &mWeights[1].front(), mWeights[1].size());
    mWeights[2].resize(mBiases[1].size() * mBiases[2].size());
    read(d, &mWeights[2].front(), mWeights[2].size());

    //std::cout << "Channels: " << mInputChannels << ", " << mBiases[0].size() << ", " << mBiases[1].size() << ", " << mBiases[2].size() << std::endl;
    //std::cout << "Activation scales: " << mActivationScales[0] << ", " << mActivationScales[1] << ", " << mActivationScales[2] << ", " << mActivationScales[3] << std::endl;
    //std::cout << "Weight scales: " << mWeightScales[0] << ", " << mWeightScales[1] << ", " << mWeightScales[2] << std::endl;

    ASSERT(d == a + length);
}

int DLRMBottomMLPPlugin::getNbOutputs() const
{
    return 1;
}

int DLRMBottomMLPPlugin::initialize()
{
    if (!mInitialized)
    {
        mDeviceWeights.resize(mWeights.size());
        for(int i = 0; i < static_cast<int>(mWeights.size()); ++i)
        {
            std::vector<int8_t> shuffledWeights = shuffleWeights(mWeights[i], mBiases[i].size(), i);
            CUDA_ASSERT(cudaMalloc(&mDeviceWeights[i], shuffledWeights.size() * sizeof(int8_t)));
            CUDA_ASSERT(cudaMemcpy(mDeviceWeights[i], &shuffledWeights.front(), shuffledWeights.size() * sizeof(int8_t), cudaMemcpyHostToDevice));
        }

        mDeviceBiases.resize(mBiases.size());
        for(int i = 0; i < static_cast<int>(mBiases.size()); ++i)
        {
            std::vector<float> scaledBiases(mBiases[i].size());
            float scale = 1.0F / mActivationScales[i + 1];
            std::transform(mBiases[i].begin(), mBiases[i].end(), scaledBiases.begin(), [=] (float x) { return x * scale; } );
            CUDA_ASSERT(cudaMalloc(&mDeviceBiases[i], scaledBiases.size() * sizeof(float)));
            CUDA_ASSERT(cudaMemcpy(mDeviceBiases[i], &scaledBiases.front(), scaledBiases.size() * sizeof(float), cudaMemcpyHostToDevice));
        }

        {
            std::string ptxStr;
            {
                std::string cudaPath;
                {
                    Dl_info info;
                    if (dladdr((void*)nvrtcCreateProgram, &info) != 0)
                    {
                        std::string token;
                        std::string cudaLibPathString;
                        //the path should be -> /path/to/cuda/lib64/libcudart.so.11.0
                        std::string s = std::string(info.dli_fname);
                        std::string delimiter = "/";
                        cudaLibPathString = s.substr(0, s.find_last_of(delimiter));
                        token = s.substr(0, cudaLibPathString.find_last_of(delimiter));
                        cudaPath = token.c_str();
                    }
                    else
                    {
                        cudaPath = getenv("CUDA_PATH");
                    }

                    if (cudaPath.empty())
                    {
                        std::cout << "Please set CUDA_PATH" << std::endl;
                        cudaPath = "/usr/local/cuda";
                    }
                }
                char includeDirOption[1024];
                int l;
                l = snprintf(includeDirOption, 1024, "-I%s/include", cudaPath.c_str());
                if (l > 1024)
                {
                    std::cout << "Cuda path too long" << std::endl;
                    ASSERT(0);
                }

                nvrtcProgram prog;
                NVRTC_ASSERT(nvrtcCreateProgram(&prog, mBottomMLPFusionKernelCode, "bottomMLPFusion.cu", 0, NULL, NULL));

                int deviceId;
                CUDA_ASSERT(cudaGetDevice(&deviceId));
                int capabilityMajor;
                CUDA_ASSERT(cudaDeviceGetAttribute(&capabilityMajor, cudaDevAttrComputeCapabilityMajor, deviceId));
                int capabilityMinor;
                CUDA_ASSERT(cudaDeviceGetAttribute(&capabilityMinor, cudaDevAttrComputeCapabilityMinor, deviceId));
                int capability = capabilityMajor * 10 + capabilityMinor;
                assert(capability >= 75);

                char threadblockSizeOption[64];
                snprintf(threadblockSizeOption, 64, "-DTHREADBLOCK_SIZE=%d", mWarpsInThreadblock * 32);
                char elementyCount0Option[64];
                snprintf(elementyCount0Option, 64, "-DELEMENT_COUNT_0=%d", mInputChannels);
                char elementyCount0PaddedOption[64];
                snprintf(elementyCount0PaddedOption, 64, "-DELEMENT_COUNT_0_PADDED=%d", (mInputChannels + 16 - 1) / 16 * 16);
                char elementyCount1Option[64];
                snprintf(elementyCount1Option, 64, "-DELEMENT_COUNT_1=%d", static_cast<int>(mBiases[0].size()));
                char elementyCount2Option[64];
                snprintf(elementyCount2Option, 64, "-DELEMENT_COUNT_2=%d", static_cast<int>(mBiases[1].size()));
                char elementyCount3Option[64];
                snprintf(elementyCount3Option, 64, "-DELEMENT_COUNT_3=%d", static_cast<int>(mBiases[2].size()));

                const char * opts[] = {
                    (capability >= 86) ? "--gpu-architecture=compute_86" : ((capability >= 80) ? "--gpu-architecture=compute_80" : "--gpu-architecture=compute_75"),
                    "-std=c++11",
                    "--extra-device-vectorization",
                    includeDirOption,
                    threadblockSizeOption,
                    elementyCount0Option,
                    elementyCount0PaddedOption,
                    elementyCount1Option,
                    elementyCount2Option,
                    elementyCount3Option};
                nvrtcResult compileNvrtcStatus = nvrtcCompileProgram(prog, 10, opts);

                size_t logSize;
                NVRTC_ASSERT(nvrtcGetProgramLogSize(prog, &logSize));
                std::vector<char> log(logSize);
                NVRTC_ASSERT(nvrtcGetProgramLog(prog, log.data()));
                std::string logStr(log.data(), logSize);
                if (compileNvrtcStatus != NVRTC_SUCCESS)
                    std::cout << "Build log: " << logStr << std::endl;
                NVRTC_ASSERT(compileNvrtcStatus);

                size_t ptxSize;
                NVRTC_ASSERT(nvrtcGetPTXSize(prog, &ptxSize));
                std::vector<char> ptx(ptxSize);
                NVRTC_ASSERT(nvrtcGetPTX(prog, ptx.data()));
                ptxStr = std::string(ptx.data(), ptxSize);

                NVRTC_ASSERT(nvrtcDestroyProgram(&prog));
            }

            CUDADRV_ASSERT(cuModuleLoadData(&mModule, ptxStr.c_str()));
            CUDADRV_ASSERT(cuModuleGetFunction(&mKernel, mModule, "fusedbottommlpkernel"));
        }

        mInitialized = true;
    }

    return 0;
}

void DLRMBottomMLPPlugin::terminate()
{
    if (mInitialized)
    {
        for(auto devicePtr: mDeviceWeights)
            CUDA_ASSERT(cudaFree(devicePtr));
        mDeviceWeights.clear();

        for(auto devicePtr: mDeviceBiases)
            CUDA_ASSERT(cudaFree(devicePtr));
        mDeviceBiases.clear();

        CUDADRV_ASSERT(cuModuleUnload(mModule));

        mInitialized = false;
    }
}

DimsExprs DLRMBottomMLPPlugin::getOutputDimensions(int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder)
{
    ASSERT(outputIndex == 0);
    ASSERT(nbInputs == 1);

    ASSERT(inputs[0].nbDims == 4);
    ASSERT(inputs[0].d[1]->isConstant());
    ASSERT(inputs[0].d[2]->isConstant());
    ASSERT(inputs[0].d[3]->isConstant());
    ASSERT(((inputs[0].d[1]->getConstantValue() + 3) / 4 * 4) == mInputChannels);
    ASSERT(inputs[0].d[2]->getConstantValue() == 1);
    ASSERT(inputs[0].d[3]->getConstantValue() == 1);

    DimsExprs outDims{4, {inputs[0].d[0], exprBuilder.constant(mBiases.back().size()), inputs[0].d[2], inputs[0].d[3]}};

    return outDims;
}

size_t DLRMBottomMLPPlugin::getWorkspaceSize(const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs, int nbOutputs) const
{
    return 0;
}

int DLRMBottomMLPPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs,
    void* const* outputs, void* workspace, cudaStream_t stream)
{
    runFusedBottomMLP(
        stream,
        ((const int8_t *)(inputs[0])),
        mDeviceWeights[0],
        mDeviceBiases[0],
        mActivationScales[0] * mWeightScales[0] / mActivationScales[1],
        mDeviceWeights[1],
        mDeviceBiases[1],
        mActivationScales[1] * mWeightScales[1] / mActivationScales[2],
        mDeviceWeights[2],
        mDeviceBiases[2],
        mActivationScales[2] * mWeightScales[2] / mActivationScales[3],
        (int8_t *)(outputs[0]),
        inputDesc[0].dims.d[0]);

    return 0;
}

void DLRMBottomMLPPlugin::runFusedBottomMLP(
    cudaStream_t stream,
    const int8_t * srcActivations,
    const int8_t * weights1,
    const float * biases1,
    float scale1,
    const int8_t * weights2,
    const float * biases2,
    float scale2,
    const int8_t * weights3,
    const float * biases3,
    float scale3,
    int8_t * dstActivations,
    int batchSize)
{
    void *args[] = {
        &srcActivations,
        &weights1,
        &biases1,
        &scale1,
        &weights2,
        &biases2,
        &scale2,
        &weights3,
        &biases3,
        &scale3,
        &dstActivations,
        &batchSize};
    const int elemsPerWarp = 16;
    const int warpsInThreadblock = 4;
    int warpCount = (batchSize + elemsPerWarp - 1) / elemsPerWarp;
    CUDADRV_ASSERT(cuLaunchKernel(
        mKernel,
        (warpCount + warpsInThreadblock - 1) / warpsInThreadblock, 1, 1,
        warpsInThreadblock * 32, 1, 1,
        0,
        stream,
        args,
        0));
}

size_t DLRMBottomMLPPlugin::getSerializationSize() const
{
    return sizeof(int) * 4 + sizeof(float) * 7 + (mBiases[0].size() + mBiases[1].size() + mBiases[2].size()) * sizeof(float)
        + (mWeights[0].size() + mWeights[1].size() + mWeights[2].size()) * sizeof(int8_t);
}

void DLRMBottomMLPPlugin::serialize(void* buffer) const
{
    char *d = reinterpret_cast<char *>(buffer);
    const char *a = d;

    write(d, mInputChannels);
    write(d, static_cast<int>(mBiases[0].size()));
    write(d, static_cast<int>(mBiases[1].size()));
    write(d, static_cast<int>(mBiases[2].size()));
    write(d, &mActivationScales.front(), mActivationScales.size());
    write(d, &mWeightScales.front(), mWeightScales.size());
    write(d, &mBiases[0].front(), mBiases[0].size());
    write(d, &mBiases[1].front(), mBiases[1].size());
    write(d, &mBiases[2].front(), mBiases[2].size());
    write(d, &mWeights[0].front(), mWeights[0].size());
    write(d, &mWeights[1].front(), mWeights[1].size());
    write(d, &mWeights[2].front(), mWeights[2].size());

    ASSERT(d == a + getSerializationSize());
}

void DLRMBottomMLPPlugin::configurePlugin(const DynamicPluginTensorDesc* in, int nbInputs, const DynamicPluginTensorDesc* out, int nbOutputs)
{
    ASSERT(in && nbInputs == 1);
    ASSERT(out && nbOutputs == 1);

    ASSERT(in[0].desc.dims.nbDims == 4);
    ASSERT(out[0].desc.dims.nbDims == 4);

    mActivationScales[0] = in[0].desc.scale;
    mActivationScales[3] = out[0].desc.scale;
}

bool DLRMBottomMLPPlugin::supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs)
{
    ASSERT(nbInputs == 1);
    ASSERT(nbOutputs == 1);

    switch (pos)
    {
        case 0:
            return ((inOut[pos].format == TensorFormat::kCHW4) && (inOut[pos].type == DataType::kINT8));
            break;
        case 1:
            return ((inOut[pos].format == TensorFormat::kCHW32) && (inOut[pos].type == inOut[0].type));
            break;
    }

    return false;
}

DataType DLRMBottomMLPPlugin::getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const 
{
    ASSERT(nbInputs == 1);
    ASSERT(index == 0);

    return inputTypes[0];
}

const char* DLRMBottomMLPPlugin::getPluginType() const { return DLRM_BOTTOM_MLP_PLUGIN_NAME; }

const char* DLRMBottomMLPPlugin::getPluginVersion() const { return DLRM_BOTTOM_MLP_PLUGIN_VERSION; }

void DLRMBottomMLPPlugin::destroy() { delete this; }

IPluginV2DynamicExt* DLRMBottomMLPPlugin::clone() const
{
    IPluginV2DynamicExt* plugin = new DLRMBottomMLPPlugin(*this);
    return plugin;
}

DLRMBottomMLPPluginCreator::DLRMBottomMLPPluginCreator()
{
    mPluginAttributes.emplace_back(PluginField("inputChannels", nullptr, PluginFieldType::kINT32, 1));

    mPluginAttributes.emplace_back(PluginField("weights0", nullptr, PluginFieldType::kFLOAT32));
    mPluginAttributes.emplace_back(PluginField("biases0", nullptr, PluginFieldType::kFLOAT32));
    mPluginAttributes.emplace_back(PluginField("dynamicRange0", nullptr, PluginFieldType::kFLOAT32, 1));

    mPluginAttributes.emplace_back(PluginField("weights1", nullptr, PluginFieldType::kFLOAT32));
    mPluginAttributes.emplace_back(PluginField("biases1", nullptr, PluginFieldType::kFLOAT32));
    mPluginAttributes.emplace_back(PluginField("dynamicRange1", nullptr, PluginFieldType::kFLOAT32, 1));

    mPluginAttributes.emplace_back(PluginField("weights2", nullptr, PluginFieldType::kFLOAT32));
    mPluginAttributes.emplace_back(PluginField("biases2", nullptr, PluginFieldType::kFLOAT32));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* DLRMBottomMLPPluginCreator::getPluginName() const
{
    return DLRM_BOTTOM_MLP_PLUGIN_NAME;
}

const char* DLRMBottomMLPPluginCreator::getPluginVersion() const
{
    return DLRM_BOTTOM_MLP_PLUGIN_VERSION;
}

const PluginFieldCollection* DLRMBottomMLPPluginCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2* DLRMBottomMLPPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    const PluginField* fields = fc->fields;

    int inputChannels = 0;
    std::vector<std::vector<float>> weights(3);
    std::vector<std::vector<float>> biases(3);
    std::vector<float> dynamicRanges(2);
 
    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "inputChannels"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            inputChannels = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "weights0"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            weights[0].resize(fields[i].length);
            memcpy(weights[0].data(), fields[i].data, fields[i].length * sizeof(float));
        }
        else if (!strcmp(attrName, "weights1"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            weights[1].resize(fields[i].length);
            memcpy(weights[1].data(), fields[i].data, fields[i].length * sizeof(float));
        }
        else if (!strcmp(attrName, "weights2"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            weights[2].resize(fields[i].length);
            memcpy(weights[2].data(), fields[i].data, fields[i].length * sizeof(float));
        }
        else if (!strcmp(attrName, "biases0"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            biases[0].resize(fields[i].length);
            memcpy(biases[0].data(), fields[i].data, fields[i].length * sizeof(float));
        }
        else if (!strcmp(attrName, "biases1"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            biases[1].resize(fields[i].length);
            memcpy(biases[1].data(), fields[i].data, fields[i].length * sizeof(float));
        }
        else if (!strcmp(attrName, "biases2"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            biases[2].resize(fields[i].length);
            memcpy(biases[2].data(), fields[i].data, fields[i].length * sizeof(float));
        }
        else if (!strcmp(attrName, "dynamicRange0"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            dynamicRanges[0] = static_cast<float>(*(static_cast<const float*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "dynamicRange1"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            dynamicRanges[1] = static_cast<float>(*(static_cast<const float*>(fields[i].data)));
        }
    }

    for(int i = 0; i < 3; ++i)
    {
        ASSERT(weights[i].size() == (biases[i].size() * (i == 0 ? inputChannels : biases[i - 1].size())));
    }

    return new DLRMBottomMLPPlugin(inputChannels, weights, biases, dynamicRanges);
}

IPluginV2* DLRMBottomMLPPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    return new DLRMBottomMLPPlugin(serialData, serialLength);
}

std::vector<int8_t> DLRMBottomMLPPlugin::shuffleWeights(
    const std::vector<int8_t>& originalWeights,
    int rows,
    int layerId)
{
    if (layerId == 2)
        assert(rows % 16 == 0);
    else
        assert(rows % 32 == 0);
    if (layerId < 2)
        assert(rows <= 512);

    int columns = originalWeights.size() / rows;

    if (layerId == 0)
        assert(columns % 4 == 0);
    else
        assert(columns % 32 == 0);
    assert(columns <= 512);

    std::vector<int8_t> shuffledWeights = originalWeights;

    // Pad input size to 16
    if ((layerId == 0) && ((columns % 16) != 0))
    {
        int columnsPadded = (columns + 16 - 1) / 16 * 16;
        std::vector<int8_t> coalescedWeights(rows * columnsPadded);
        for(int row = 0; row < rows; ++row)
        {
            std::copy(
                shuffledWeights.begin() + row * columns,
                shuffledWeights.begin() + (row + 1) * columns,
                coalescedWeights.begin() + row * columnsPadded);
        }
        shuffledWeights = coalescedWeights;
        columns = columnsPadded;
    }

    // Reshuffle weights so that 2 adjacent accums (int2 each) are nicely aligned - store adjacent output elements
    {
        std::vector<int8_t> coalescedWeights(shuffledWeights.size());
        for(int dstRow = 0; dstRow < rows; ++dstRow)
        {
            int srcRow = (dstRow & ~14) | (((dstRow >> 1) & 3) << 2) | (((dstRow >> 3) & 1) << 1);
            std::copy(
                shuffledWeights.begin() + srcRow * columns,
                shuffledWeights.begin() + (srcRow + 1) * columns,
                coalescedWeights.begin() + dstRow * columns);
        }
        shuffledWeights = coalescedWeights;
    }

    {
        // Shuffle weights so that the kernel could load them in a fully coalesced manner 
        std::vector<int8_t> coalescedWeights(shuffledWeights.size());
        auto dstIt = coalescedWeights.begin();
        for(int fragmentId = 0; fragmentId < rows / 8; fragmentId += 2)
            for(int k = 0; k < columns / 16; ++k)
                for(int i = 0; i < 16; ++i)
                {
                    int srcOffset = fragmentId * 8 * columns + k * 16 + i * columns;
                    dstIt = std::copy(shuffledWeights.begin() + srcOffset, shuffledWeights.begin() + srcOffset + 16, dstIt);
                }
        shuffledWeights = coalescedWeights;
    }

    // Shuffle weights to allow int2 load
    {
        std::vector<int8_t> coalescedWeights(shuffledWeights.size());
        auto dstIt = coalescedWeights.begin();
        for(int k = 0; k < static_cast<int>(shuffledWeights.size()) / 256; ++k)
        {
            for (int i = 0; i < 32; ++i)
            {
                dstIt = std::copy(shuffledWeights.begin() + (k * 256 + i * 4), shuffledWeights.begin() + (k * 256 + i * 4 + 4), dstIt);
                dstIt = std::copy(shuffledWeights.begin() + (k * 256 + 128 + i * 4), shuffledWeights.begin() + (k * 256 + 128 + i * 4 + 4), dstIt);
            }
        }
        shuffledWeights = coalescedWeights;
    }

    // Shuffle weights to allow int4 load
    {
        std::vector<int8_t> coalescedWeights(shuffledWeights.size());
        auto dstIt = coalescedWeights.begin();
        for(int k = 0; k < static_cast<int>(shuffledWeights.size()) / 512; ++k)
        {
            for (int i = 0; i < 64; ++i)
            {
                dstIt = std::copy(shuffledWeights.begin() + (k * 512 + i * 4), shuffledWeights.begin() + (k * 512 + i * 4 + 4), dstIt);
                dstIt = std::copy(shuffledWeights.begin() + (k * 512 + 256 + i * 4), shuffledWeights.begin() + (k * 512 + 256 + i * 4 + 4), dstIt);
            }
        }
        shuffledWeights = coalescedWeights;
    }

    return shuffledWeights;
}
