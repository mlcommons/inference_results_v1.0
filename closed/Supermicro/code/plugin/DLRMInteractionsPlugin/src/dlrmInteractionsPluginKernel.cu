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

#include "dlrmInteractionsPluginKernel.h"
#include "dlrmHelper.h"

#include <vector>
#include <iostream>
#include <algorithm>
#include <cassert>
#include <math.h>

#include <cuda_fp16.h>
#include <mma.h>
#if __CUDA_ARCH__ >= 800
#include <cuda_pipeline.h>
#endif

using namespace nvcuda;

struct __align__(8) half4
{
    __host__ __device__ half4() {}
    __host__ __device__ half4(const half4& other) : v(other.v) {}
    __host__ __device__ half4& operator =(const half4& other) { v = other.v; return *this; }
    union
    {
        half2 vals[2];
        int2 v;
    };
};

__forceinline__ __device__ half2 make_zero_half2()
{
    return make_half2((half)0.0F, (half)0.0F);
}

__forceinline__ __device__ half4 fromPackedInt8(unsigned int packedVals, __half scale)
{
    half4 res;
    res.vals[0].x = __short2half_rn((int8_t)(packedVals & 0xFF));
    res.vals[0].y = __short2half_rn((int8_t)((packedVals >> 8) & 0xFF));
    res.vals[1].x = __short2half_rn((int8_t)((packedVals >> 16) & 0xFF));
    res.vals[1].y = __short2half_rn((int8_t)((packedVals >> 24) & 0xFF));
    res.vals[0] = __hmul2(res.vals[0], __half2(scale, scale));
    res.vals[1] = __hmul2(res.vals[1], __half2(scale, scale));
    return res;
}

__forceinline__ __device__ unsigned int toPackedInt8(half4 v, __half scale)
{
    v.vals[0] = __hmul2(v.vals[0], __half2(scale, scale));
    v.vals[1] = __hmul2(v.vals[1], __half2(scale, scale));
    return (max(min(__half2int_rn(v.vals[0].x), 127), -127) & 0xFF)
        | ((max(min(__half2int_rn(v.vals[0].y), 127), -127) & 0xFF) << 8)
        | ((max(min(__half2int_rn(v.vals[1].x), 127), -127) & 0xFF) << 16)
        | ((max(min(__half2int_rn(v.vals[1].y), 127), -127) & 0xFF) << 24);
}

__forceinline__ __device__ int8_t toInt8(__half v, __half scale)
{
    return max(min(__half2int_rn(__hmul(v, scale)), 127), -127);
}

struct EmbeddingScales
{
    __half vals[65];
};

template<bool REMAP_INDICES, typename InOutType, int WARPS_PER_BLOCK, int THREADBLOCK_SIZE, int M_BLOCKS, int K_BLOCKS, int SMEM_STRIDE, int SMEM_STRIDE_ACC, int LOADS_IN_FLIGHT, int STORES_IN_FLIGHT>
__launch_bounds__(THREADBLOCK_SIZE)
__global__ void batchedHsyrk(
    const InOutType * __restrict denseInput,
    const int * __restrict sparseInput,
    const int8_t * __restrict embeddings,
    const int8_t * __restrict embeddingsHost,
    const int * __restrict indexRemapping,
    const int * __restrict tableOffsets,
    EmbeddingScales scales,
    InOutType * __restrict output,
    int m,
    int batchSize,
    int embeddingRowsOnDevice,
    int outputSizePadded,
    int smemElemsPerWarp,
    int smemRowsPerWarp)
{
    int warpId = (threadIdx.x >> 5);
    int sampleId = blockIdx.x * WARPS_PER_BLOCK + warpId;
    if (sampleId >= batchSize)
        return;
    int laneId = threadIdx.x & 31;

    extern __shared__ half shmem_dynamic[];
    half * shmem = shmem_dynamic + (warpId * smemElemsPerWarp);

    // Load dense feature
    const InOutType * sampleDenseInput = denseInput + (K_BLOCKS * 16) * sampleId;
    half4 denseFeatures;
    if (laneId < K_BLOCKS * 4)
    {
        if (sizeof(InOutType) == 2)
        {
            denseFeatures.v = ((int2 *)sampleDenseInput)[laneId];
        }
        else
        {
            float4 denseFeaturesFP32 = ((float4 *)sampleDenseInput)[laneId];
            denseFeatures.vals[0].x = __float2half_rn(denseFeaturesFP32.x);
            denseFeatures.vals[0].y = __float2half_rn(denseFeaturesFP32.y);
            denseFeatures.vals[1].x = __float2half_rn(denseFeaturesFP32.z);
            denseFeatures.vals[1].y = __float2half_rn(denseFeaturesFP32.w);
        }
    }

    // Load embeddings for sparse features
    int sparseFeatures = m - 1;
    const int * sampleSparseInput = sparseInput + sparseFeatures * sampleId;
#pragma unroll 1
    for(int baseSparseFeature = 0; baseSparseFeature < sparseFeatures; baseSparseFeature += 32)
    {
        half * shmem_sparse = shmem + SMEM_STRIDE + baseSparseFeature * SMEM_STRIDE;
        int rowsToLoad = min(32, sparseFeatures - baseSparseFeature);
        int index = 0;
        if (laneId < rowsToLoad)
        {
            int embeddingIndex = sampleSparseInput[baseSparseFeature + laneId] + tableOffsets[baseSparseFeature + laneId];
            index = REMAP_INDICES ? indexRemapping[embeddingIndex] : embeddingIndex;
        }
        const int8_t * embeddingsDistributed = index < embeddingRowsOnDevice ?
            embeddings + (long long)index * (K_BLOCKS * 16) : embeddingsHost + (long long)(index - embeddingRowsOnDevice) * (K_BLOCKS * 16);

        int i = 0;
#pragma unroll 1
        for(; i < (rowsToLoad / LOADS_IN_FLIGHT * LOADS_IN_FLIGHT); i += LOADS_IN_FLIGHT)
        {
            unsigned int vals[LOADS_IN_FLIGHT];
            for(int j = 0; j < LOADS_IN_FLIGHT; ++j)
            {
                unsigned long long ptr = __shfl_sync(0xffffffff, *((unsigned long long *)&embeddingsDistributed), i + j);
                const unsigned int * currentEmbeddings = *((unsigned int **)&ptr);
                if (laneId < K_BLOCKS * 4)
                    vals[j] = __ldg(currentEmbeddings + laneId);
            }
            for(int j = 0; j < LOADS_IN_FLIGHT; ++j)
            {
                if (laneId < K_BLOCKS * 4)
                    ((int2 *)(shmem_sparse + (i + j) * SMEM_STRIDE))[laneId] = fromPackedInt8(vals[j], scales.vals[baseSparseFeature + i + j]).v;
            }
        }
        for(; i < rowsToLoad; ++i)
        {
            unsigned long long ptr = __shfl_sync(0xffffffff, *((unsigned long long *)&embeddingsDistributed), i);
            const unsigned int * currentEmbeddings = *((unsigned int **)&ptr);
            if (laneId < K_BLOCKS * 4)
                ((int2 *)(shmem_sparse + i * SMEM_STRIDE))[laneId] = fromPackedInt8(__ldg(currentEmbeddings + laneId), scales.vals[baseSparseFeature + i]).v;
        }
    }

    if (laneId < K_BLOCKS * 4)
        ((int2 *)(shmem))[laneId] = denseFeatures.v;

    __syncwarp();

    wmma::fragment<wmma::accumulator, 16, 16, 16, half> acc[M_BLOCKS][M_BLOCKS];

    for (int i = 0; i < M_BLOCKS; i++)
        for (int j = 0; j < M_BLOCKS; j++)
            wmma::fill_fragment(acc[i][j], (half)0);

    for (int k_step = 0; k_step < K_BLOCKS; k_step++)
    {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a[M_BLOCKS];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b[M_BLOCKS];
        for (int j = 0; j < M_BLOCKS; j++)
        {
            int base_row = (j < M_BLOCKS - 1) ? j * 16 : smemRowsPerWarp - 16;
            const half * tile_ptr = shmem + (base_row * SMEM_STRIDE + k_step * 16);
            wmma::load_matrix_sync(a[j], tile_ptr, SMEM_STRIDE);
            wmma::load_matrix_sync(b[j], tile_ptr, SMEM_STRIDE);
        }
        for (int i = 0; i < M_BLOCKS; i++)
            for (int j = 0; j < M_BLOCKS; j++)
                if (j <= i)
                    wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
    }

    for (int i = 0; i < M_BLOCKS; i++)
        for (int j = 0; j < M_BLOCKS; j++)
            if (j <= i)
            {
                half * tile_ptr = shmem + (i * 16 * SMEM_STRIDE_ACC + j * 16);
                wmma::store_matrix_sync(tile_ptr, acc[i][j], SMEM_STRIDE_ACC, wmma::mem_row_major);
            }

    InOutType * sampleOutput = output + outputSizePadded * sampleId;
    // Store dense features
    if (laneId < K_BLOCKS * 4)
        if (sizeof(InOutType) == 2)
        {
            ((int2 *)sampleOutput)[laneId] = denseFeatures.v;
        }
        else
        {
            float4 denseFeaturesFP32;
            denseFeaturesFP32.x = __half2float(denseFeatures.vals[0].x);
            denseFeaturesFP32.y = __half2float(denseFeatures.vals[0].y);
            denseFeaturesFP32.z = __half2float(denseFeatures.vals[1].x);
            denseFeaturesFP32.w = __half2float(denseFeatures.vals[1].y);
            ((float4 *)sampleOutput)[laneId] = denseFeaturesFP32;
        }
    sampleOutput += (K_BLOCKS * 16);
    // Store interactions
    int maxInteractionsOutput = outputSizePadded - (K_BLOCKS * 16);
    int lastRowBlockOffset = M_BLOCKS * 16 - smemRowsPerWarp;

    int flatId = laneId;
    int minElemsToStore = maxInteractionsOutput / 32;
#pragma unroll 1
    for(int i = 0; i < minElemsToStore / STORES_IN_FLIGHT; ++i, flatId += 32 * STORES_IN_FLIGHT)
    {
        half vals[STORES_IN_FLIGHT];
        for(int j = 0; j < STORES_IN_FLIGHT; ++j)
            vals[j] = (half)0.0F;
        for(int j = 0; j < STORES_IN_FLIGHT; ++j)
        {
            int localFlatId = flatId + j * 32;
            int logicalRow = __float2int_rz(sqrtf(2.0F * localFlatId + 0.25F) + 0.5F);
            if (logicalRow < m)
            {
                int logicalColumn = localFlatId - (logicalRow * (logicalRow - 1) / 2);
                int srcRow = logicalRow;
                if (srcRow >= ((M_BLOCKS - 1) * 16))
                    srcRow += lastRowBlockOffset;
                int srcColumn = logicalColumn;
                if (srcColumn >= ((M_BLOCKS - 1) * 16))
                    srcColumn += lastRowBlockOffset;
                vals[j] = shmem[srcRow * SMEM_STRIDE_ACC + srcColumn];
            }
        }
        for(int j = 0; j < STORES_IN_FLIGHT; ++j)
            if (sizeof(InOutType) == 2)
                sampleOutput[flatId + j * 32] = vals[j];
            else
                sampleOutput[flatId + j * 32] = __half2float(vals[j]);
        }
    for(; flatId < maxInteractionsOutput; flatId += 32)
    {
        half val = (half)0.0F;
        int logicalRow = __float2int_rz(sqrtf(2.0F * flatId + 0.25F) + 0.5F);
        if (logicalRow < m)
        {
            int logicalColumn = flatId - (logicalRow * (logicalRow - 1) / 2);
            int srcRow = logicalRow;
            if (srcRow >= ((M_BLOCKS - 1) * 16))
                srcRow += lastRowBlockOffset;
            int srcColumn = logicalColumn;
            if (srcColumn >= ((M_BLOCKS - 1) * 16))
                srcColumn += lastRowBlockOffset;
            val = shmem[srcRow * SMEM_STRIDE_ACC + srcColumn];
        }
        if (sizeof(InOutType) == 2)
            sampleOutput[flatId] = val;
        else
            sampleOutput[flatId] = __half2float(val);
    }
}

union int8pack
{
    int v;
    int8_t t[4];
};

__forceinline__ __device__ unsigned int rescalePackedInt8(unsigned int packedVals, float scale)
{
    int8pack in;
    in.v = packedVals;
#if __CUDA_ARCH__ >= 860
    int res;
    int xi = __float2int_rn(max(__int2float_rn(in.t[0]) * scale, -127.0F));
    int yi = __float2int_rn(max(__int2float_rn(in.t[1]) * scale, -127.0F));
    int zi = __float2int_rn(max(__int2float_rn(in.t[2]) * scale, -127.0F));
    int wi = __float2int_rn(max(__int2float_rn(in.t[3]) * scale, -127.0F));
    asm(" { .reg .u32 t1; cvt.pack.sat.s8.s32.b32 t1, %1, %2, 0; cvt.pack.sat.s8.s32.b32 %0, %3, %4, t1; } " : "=r"(res) : "r"(wi), "r"(zi), "r"(yi), "r"(xi));
    return res;
#else
    int8pack res;
    res.t[0] = __float2int_rn(max(min(__int2float_rn(in.t[0]) * scale, 127.0F), -127.0F));
    res.t[1] = __float2int_rn(max(min(__int2float_rn(in.t[1]) * scale, 127.0F), -127.0F));
    res.t[2] = __float2int_rn(max(min(__int2float_rn(in.t[2]) * scale, 127.0F), -127.0F));
    res.t[3] = __float2int_rn(max(min(__int2float_rn(in.t[3]) * scale, 127.0F), -127.0F));
    return res.v;
#endif
}

__forceinline__ __device__ float slowIntToFloat(int v)
{
    return __int2float_rn(v);
}

__forceinline__ __device__ float fastIntToFloat(int v)
{
    return __int_as_float(v) - 12582912.0F;
}

__forceinline__ __device__ int8_t slowFloatToInt8(float v)
{
    return __float2int_rn(max(min(v, 127.0F), -127.0F));
}

__forceinline__ __device__ int8_t fastFloatToInt8(float v)
{
    return __float_as_int(min(max(v + 12582912.0F, 12582785.0F), 12583039.0F));
}

template<int SAMPLES_PER_WARP, bool OUTPUT_INTERLEAVED, bool REMAP_INDICES, int WARPS_PER_BLOCK, int THREADBLOCK_SIZE, int M_BLOCKS, int K_BLOCKS, int SMEM_STRIDE, int LOADS_IN_FLIGHT, int LOADS_IN_FLIGHT_LDGSTS>
__launch_bounds__(THREADBLOCK_SIZE)
__global__ void batchedIsyrk(
    const int8_t * __restrict denseInput,
    const int * __restrict sparseInput,
    const int8_t * __restrict embeddings,
    const int8_t * __restrict embeddingsHost,
    const int * __restrict indexRemapping,
    const int * __restrict tableOffsets,
    float denseConversionScale,
    const float * __restrict scales,
    int8_t * __restrict output,
    int m,
    int batchSize,
    int embeddingRowsOnDevice,
    int outputSizePadded,
    int smemElemsPerSample,
    int smemRowsPerSample)
{
#if __CUDA_ARCH__ >= 750
    int warpId = (threadIdx.x >> 5);
    int baseSampleId = blockIdx.x * (WARPS_PER_BLOCK * SAMPLES_PER_WARP) + warpId * SAMPLES_PER_WARP;
    if (baseSampleId >= batchSize)
        return;
    int laneId = threadIdx.x & 31;

    extern __shared__ signed char shmem_dynamic_sc[];
    signed char * shmem = shmem_dynamic_sc + (warpId * smemElemsPerSample * SAMPLES_PER_WARP);

    // Load dense feature
    const unsigned int * sampleDenseInput = (const unsigned int *)(denseInput + (K_BLOCKS * 16) * baseSampleId);
    unsigned int denseFeatures[SAMPLES_PER_WARP];
#if __CUDA_ARCH__ >= 800
    if (laneId < K_BLOCKS * SAMPLES_PER_WARP)
    {
        int localSampleId = laneId / K_BLOCKS;
        int elemId = laneId % K_BLOCKS;
        unsigned int smem_offset = __cvta_generic_to_shared(shmem) + smemElemsPerSample * localSampleId + elemId * 16;
        const int4 * src = ((const int4 *)sampleDenseInput) + laneId;
        asm("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(smem_offset), "l"(src));
    }
    __pipeline_commit();

    // Load embeddings for sparse features
    int sparseFeatures = m - 1;
    const int * sampleSparseInput = sparseInput + sparseFeatures * baseSampleId;
#pragma unroll 1
    for(int baseSparseFeature = 0; baseSparseFeature < sparseFeatures; baseSparseFeature += 32)
    {
        signed char * shmem_sparse = shmem + SMEM_STRIDE + baseSparseFeature * SMEM_STRIDE;
        int rowsToLoad = min(32, sparseFeatures - baseSparseFeature);
        int index[SAMPLES_PER_WARP];
#pragma unroll
        for(int sampleId = 0; sampleId < SAMPLES_PER_WARP; ++sampleId)
            index[sampleId] = 0;
        if (laneId < rowsToLoad)
        {
            int tableOffset = tableOffsets[baseSparseFeature + laneId];
            int embeddingIndex[SAMPLES_PER_WARP];
            for(int sampleId = 0; sampleId < SAMPLES_PER_WARP; ++sampleId)
                embeddingIndex[sampleId] = sampleSparseInput[sparseFeatures * sampleId + baseSparseFeature + laneId] + tableOffset;
            for(int sampleId = 0; sampleId < SAMPLES_PER_WARP; ++sampleId)
                index[sampleId] = REMAP_INDICES ? indexRemapping[embeddingIndex[sampleId]] : embeddingIndex[sampleId];
        }
        const int8_t * embeddingsDistributed[SAMPLES_PER_WARP];
#pragma unroll
        for(int sampleId = 0; sampleId < SAMPLES_PER_WARP; ++sampleId)
            embeddingsDistributed[sampleId] = index[sampleId] < embeddingRowsOnDevice ?
                embeddings + (long long)index[sampleId] * (K_BLOCKS * 16) : embeddingsHost + (long long)(index[sampleId] - embeddingRowsOnDevice) * (K_BLOCKS * 16);
        int totalLoads = (rowsToLoad * (K_BLOCKS * 1) + 31) / 32;
        unsigned int smem_offset = __cvta_generic_to_shared(shmem_sparse);
        int flatId = laneId;
#pragma unroll LOADS_IN_FLIGHT_LDGSTS
        for(int i = 0; i < totalLoads; ++i, flatId += 32)
        {
            int rowId = flatId / (K_BLOCKS * 1);
            int elemId = flatId % (K_BLOCKS * 1);
            unsigned int current_base_smem_offset = smem_offset + rowId * SMEM_STRIDE + elemId * 16;
#pragma unroll
            for(int sampleId = 0; sampleId < SAMPLES_PER_WARP; ++sampleId)
            {
                unsigned long long ptr = __shfl_sync(0xffffffff, *((unsigned long long *)(embeddingsDistributed + sampleId)), rowId);
                if (rowId < rowsToLoad)
                {
                    const unsigned int * currentEmbeddings = *((unsigned int **)&ptr);
                    const int4 * src = ((const int4 *)currentEmbeddings) + elemId;
                    unsigned int current_smem_offset = current_base_smem_offset + smemElemsPerSample * sampleId;
                    asm("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(current_smem_offset), "l"(src));
                }
            }
        }
    }
    __pipeline_commit();

    __pipeline_wait_prior(1);
    if (laneId < K_BLOCKS * 4)
#pragma unroll
        for(int sampleId = 0; sampleId < SAMPLES_PER_WARP; ++sampleId)
            denseFeatures[sampleId] = ((unsigned int *)(shmem + smemElemsPerSample * sampleId))[laneId];

    __pipeline_wait_prior(0);
#else
    if (laneId < K_BLOCKS * 4)
#pragma unroll
        for(int sampleId = 0; sampleId < SAMPLES_PER_WARP; ++sampleId)
            denseFeatures[sampleId] = sampleDenseInput[sampleId * (K_BLOCKS * 4) + laneId];

    // Load embeddings for sparse features
    int sparseFeatures = m - 1;
    const int * sampleSparseInput = sparseInput + sparseFeatures * baseSampleId;
#pragma unroll 1
    for(int baseSparseFeature = 0; baseSparseFeature < sparseFeatures; baseSparseFeature += 32)
    {
        signed char * shmem_sparse = shmem + SMEM_STRIDE + baseSparseFeature * SMEM_STRIDE;
        int rowsToLoad = min(32, sparseFeatures - baseSparseFeature);
        int index[SAMPLES_PER_WARP];
#pragma unroll
        for(int sampleId = 0; sampleId < SAMPLES_PER_WARP; ++sampleId)
            index[sampleId] = 0;
        if (laneId < rowsToLoad)
        {
            int tableOffset = tableOffsets[baseSparseFeature + laneId];
            int embeddingIndex[SAMPLES_PER_WARP];
#pragma unroll
            for(int sampleId = 0; sampleId < SAMPLES_PER_WARP; ++sampleId)
                embeddingIndex[sampleId] = sampleSparseInput[sparseFeatures * sampleId + baseSparseFeature + laneId] + tableOffset;
#pragma unroll
            for(int sampleId = 0; sampleId < SAMPLES_PER_WARP; ++sampleId)
                index[sampleId] = REMAP_INDICES ? indexRemapping[embeddingIndex[sampleId]] : embeddingIndex[sampleId];
        }
        const int8_t * embeddingsDistributed[SAMPLES_PER_WARP];
        for(int sampleId = 0; sampleId < SAMPLES_PER_WARP; ++sampleId)
            embeddingsDistributed[sampleId] = index[sampleId] < embeddingRowsOnDevice ?
                embeddings + (long long)index[sampleId] * (K_BLOCKS * 16) : embeddingsHost + (long long)(index[sampleId] - embeddingRowsOnDevice) * (K_BLOCKS * 16);

        int i = 0;
#pragma unroll 1
        for(; i < (rowsToLoad / LOADS_IN_FLIGHT * LOADS_IN_FLIGHT); i += LOADS_IN_FLIGHT)
        {
            unsigned int vals[LOADS_IN_FLIGHT][SAMPLES_PER_WARP];
#pragma unroll
            for(int j = 0; j < LOADS_IN_FLIGHT; ++j)
            {
#pragma unroll
                for(int sampleId = 0; sampleId < SAMPLES_PER_WARP; ++sampleId)
                {
                    unsigned long long ptr = __shfl_sync(0xffffffff, *((unsigned long long *)(embeddingsDistributed + sampleId)), i + j);
                    const unsigned int * currentEmbeddings = *((unsigned int **)&ptr);
                    if (laneId < K_BLOCKS * 4)
                        vals[j][sampleId] = __ldg(currentEmbeddings + laneId);
                }
            }
#pragma unroll
            for(int j = 0; j < LOADS_IN_FLIGHT; ++j)
            {
#pragma unroll
                for(int sampleId = 0; sampleId < SAMPLES_PER_WARP; ++sampleId)
                {
                    if (laneId < K_BLOCKS * 4)
                        ((unsigned int *)(shmem_sparse + smemElemsPerSample * sampleId + (i + j) * SMEM_STRIDE))[laneId] = vals[j][sampleId];
                }
            }
        }
        for(; i < rowsToLoad; ++i)
        {
#pragma unroll
            for(int sampleId = 0; sampleId < SAMPLES_PER_WARP; ++sampleId)
            {
                unsigned long long ptr = __shfl_sync(0xffffffff, *((unsigned long long *)(embeddingsDistributed + sampleId)), i);
                const unsigned int * currentEmbeddings = *((unsigned int **)&ptr);
                if (laneId < K_BLOCKS * 4)
                    ((unsigned int *)(shmem_sparse + smemElemsPerSample * sampleId + i * SMEM_STRIDE))[laneId] = __ldg(currentEmbeddings + laneId);
            }
        }
    }

    if (laneId < K_BLOCKS * 4)
#pragma unroll
        for(int sampleId = 0; sampleId < SAMPLES_PER_WARP; ++sampleId)
            ((unsigned int *)(shmem + smemElemsPerSample * sampleId))[laneId] = denseFeatures[sampleId];

    __syncwarp();
#endif

    unsigned int smem_offset;
#if __CUDACC_VER_MAJOR__ >= 11
    smem_offset = __cvta_generic_to_shared(shmem);
#else
    asm("{\n .reg .u64 t1;\n cvta.to.shared.u64 t1, %1;\n cvt.u32.u64 %0, t1;\n }" : "=r"(smem_offset) : "l"(shmem));
#endif

    int2 acc[M_BLOCKS][M_BLOCKS][SAMPLES_PER_WARP];
#pragma unroll
    for (int i = 0; i < M_BLOCKS; i++)
#pragma unroll
        for (int j = 0; j < M_BLOCKS; j++)
#pragma unroll
            for(int sampleId = 0; sampleId < SAMPLES_PER_WARP; ++sampleId)
                acc[i][j][sampleId] = make_int2(0x4B400000, 0x4B400000);

#pragma unroll
    for(int sampleId = 0; sampleId < SAMPLES_PER_WARP; ++sampleId)
#pragma unroll
        for (int k_step = 0; k_step < K_BLOCKS; k_step++)
        {
            int a[M_BLOCKS];
#pragma unroll
            for (int j = 0; j < M_BLOCKS; j++)
            {
                int base_row = (j < M_BLOCKS - 1) ? j * 8 : smemRowsPerSample - 8;
                int tile_smem_offset = smem_offset + (base_row + (laneId & 7)) * SMEM_STRIDE + k_step * 16 + sampleId * smemElemsPerSample;
                asm("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];" : "=r"(a[j]) : "r"(tile_smem_offset));
            }

#pragma unroll
            for (int i = 0; i < M_BLOCKS; i++)
#pragma unroll
                for (int j = 0; j < M_BLOCKS; j++)
                    if (j <= i)
                        asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};" : "+r"(acc[i][j][sampleId].x), "+r"(acc[i][j][sampleId].y) : "r"(a[i]), "r"(a[j]));
        }

    int8_t * sampleOutput;
    if (OUTPUT_INTERLEAVED)
        sampleOutput = output + (outputSizePadded * (baseSampleId & ~1) + (baseSampleId & 1) * 32);
    else
        sampleOutput = output + outputSizePadded * baseSampleId;
    // Store dense features
    const int OUTPUT_SAMPLE_OFFSET_INT = OUTPUT_INTERLEAVED ? 8 : (outputSizePadded / 4);
    if (laneId < K_BLOCKS * 4)
#pragma unroll
        for(int sampleId = 0; sampleId < SAMPLES_PER_WARP; ++sampleId)
            ((unsigned int *)sampleOutput)[(OUTPUT_INTERLEAVED ? (laneId & 7) | ((laneId & ~7) << 1) : laneId) + OUTPUT_SAMPLE_OFFSET_INT * sampleId] =
                rescalePackedInt8(denseFeatures[sampleId], denseConversionScale);

    int logicalRows[M_BLOCKS];
    int rowOffsets[M_BLOCKS];
    float rowScales[M_BLOCKS];
#pragma unroll
    for (int i = 0; i < M_BLOCKS; i++)
    {
        if (i < M_BLOCKS - 1)
            logicalRows[i] = i * 8 + (laneId >> 2);
        else
        {
            logicalRows[i] = (smemRowsPerSample - 8) + (laneId >> 2);
            if (logicalRows[i] < (M_BLOCKS - 1) * 8)
                logicalRows[i] = m;
        }
        rowOffsets[i] = logicalRows[i] * (logicalRows[i] - 1) / 2 + (K_BLOCKS * 16);
        if (logicalRows[i] < m)
            rowScales[i] = scales[logicalRows[i]];
    }
    int logicalCols[M_BLOCKS];
    float2 colScales[M_BLOCKS];
#pragma unroll
    for (int j = 0; j < M_BLOCKS; j++)
    {
        if (j < M_BLOCKS - 1)
            logicalCols[j] = j * 8 + ((laneId & 3) << 1);
        else
        {
            logicalCols[j] = (smemRowsPerSample - 8) + ((laneId & 3) << 1);
            if (logicalCols[j] < (M_BLOCKS - 1) * 8)
                logicalCols[j] = m;
        }
        if (logicalCols[j] < m)
            colScales[j] = *((const float2 *)(scales + logicalCols[j]));
    }

    const int OUTPUT_SAMPLE_OFFSET_BYTE = OUTPUT_INTERLEAVED ? 32 : outputSizePadded;
#pragma unroll
    for (int i = 0; i < M_BLOCKS; i++) 
#pragma unroll
        for (int j = 0; j < M_BLOCKS; j++)
            if (j <= i)
            {
                if ((i < M_BLOCKS - 1) || (logicalRows[i] < m))
                {
                    int flatOffset = rowOffsets[i] + logicalCols[j];
                    if ((j < i) || (logicalCols[j] < logicalRows[i]))
                    {
#pragma unroll
                        for(int sampleId = 0; sampleId < SAMPLES_PER_WARP; ++sampleId)
                            sampleOutput[(OUTPUT_INTERLEAVED ? (flatOffset & 31) | ((flatOffset & ~31) << 1) : flatOffset) + sampleId * OUTPUT_SAMPLE_OFFSET_BYTE] =
                                fastFloatToInt8(fastIntToFloat(acc[i][j][sampleId].x) * (rowScales[i] * colScales[j].x));
                    }
                    flatOffset += 1;
                    if ((j < i) || (logicalCols[j] + 1 < logicalRows[i]))
#pragma unroll
                        for(int sampleId = 0; sampleId < SAMPLES_PER_WARP; ++sampleId)
                            sampleOutput[(OUTPUT_INTERLEAVED ? (flatOffset & 31) | ((flatOffset & ~31) << 1) : flatOffset) + sampleId * OUTPUT_SAMPLE_OFFSET_BYTE] =
                                fastFloatToInt8(fastIntToFloat(acc[i][j][sampleId].y) * (rowScales[i] * colScales[j].y));
                }
            }

    int storedInteractions = m * (m - 1) / 2 + (K_BLOCKS * 16);
    for(int flatOffset = laneId + storedInteractions; flatOffset < outputSizePadded; flatOffset += 32)
#pragma unroll
        for(int sampleId = 0; sampleId < SAMPLES_PER_WARP; ++sampleId)
            sampleOutput[(OUTPUT_INTERLEAVED ? (flatOffset & 31) | ((flatOffset & ~31) << 1) : flatOffset) + sampleId * OUTPUT_SAMPLE_OFFSET_BYTE] = 0;
#else
    assert(0);
#endif
}

#define runFp16Kernel(REMAP_INDICES_CONST,InOutType_CONST,M_BLOCKS_CONST,K_BLOCKS_CONST) \
    batchedHsyrk<REMAP_INDICES_CONST,InOutType_CONST,warps_per_threadblock,threadblock_size,M_BLOCKS_CONST,K_BLOCKS_CONST,(K_BLOCKS_CONST*16+SKEW_HALF),(M_BLOCKS_CONST*16+SKEW_HALF_ACC),LOADS_IN_FLIGHT,STORES_IN_FLIGHT><<<(batchSize+warps_per_threadblock-1)/warps_per_threadblock,threadblock_size,warps_per_threadblock*smem_elems_per_warp*sizeof(__half),stream>>>( \
        (const InOutType_CONST *)denseInput, sparseInput, (const int8_t *)embeddings, (const int8_t *)embeddingsHost, indexRemapping, \
        tableOffsets, scales, (InOutType_CONST *)output, m, batchSize, embeddingRowsOnDevice, outputSizePadded, smem_elems_per_warp, smem_rows_per_warp);
#define runFp16Kernel2(REMAP_INDICES,InOutType_CONST,M_BLOCKS_CONST,K_BLOCKS_CONST) \
    if (REMAP_INDICES) \
    { \
        runFp16Kernel(true,InOutType_CONST,M_BLOCKS_CONST,K_BLOCKS_CONST); \
    } \
    else \
    { \
        runFp16Kernel(false,InOutType_CONST,M_BLOCKS_CONST,K_BLOCKS_CONST); \
    }
#define runFp16Kernel3(REMAP_INDICES,InOutType,M_BLOCKS_CONST,K_BLOCKS_CONST) \
    switch (InOutType) \
    { \
        case InOutDataType::FLOAT: \
        runFp16Kernel2(REMAP_INDICES,float,M_BLOCKS_CONST,K_BLOCKS_CONST); \
            break; \
        case InOutDataType::HALF: \
        runFp16Kernel2(REMAP_INDICES,__half,M_BLOCKS_CONST,K_BLOCKS_CONST); \
            break; \
        default: \
            ASSERT(0); \
            break; \
    }
#define runFp16Kernel4(REMAP_INDICES,InOutType,M_BLOCKS,K_BLOCKS_CONST) \
    switch (M_BLOCKS) \
    { \
        case 2: \
            runFp16Kernel3(REMAP_INDICES,InOutType,2,K_BLOCKS_CONST); \
            break; \
        case 4: \
            runFp16Kernel3(REMAP_INDICES,InOutType,4,K_BLOCKS_CONST); \
            break; \
        default: \
            ASSERT(0); \
            break; \
    }
#define runFp16Kernel5(REMAP_INDICES,InOutType,M_BLOCKS,K_BLOCKS) \
    switch (K_BLOCKS) \
    { \
        case 2: \
            runFp16Kernel4(REMAP_INDICES,InOutType,M_BLOCKS,2); \
            break; \
        case 4: \
            runFp16Kernel4(REMAP_INDICES,InOutType,M_BLOCKS,4); \
            break; \
        case 6: \
            runFp16Kernel4(REMAP_INDICES,InOutType,M_BLOCKS,6); \
            break; \
        case 8: \
            runFp16Kernel4(REMAP_INDICES,InOutType,M_BLOCKS,8); \
            break; \
        default: \
            ASSERT(0); \
            break; \
    }

int runBatchedSyrk(
    cudaStream_t stream,
    int m,
    int k,
    int batchSize,
    int embeddingRowsOnDevice,
    int outputSizePadded,
    const void * denseInput,
    const int * sparseInput,
    const void * embeddings,
    const void * embeddingsHost,
    const int * indexRemapping,
    const int * tableOffsets,
    const std::vector<float>& embeddingsScales,
    void * output,
    InOutDataType inOutType)
{
    ASSERT((k % 32) == 0);
    ASSERT(k <= 128);
    ASSERT(m <= 64);
    ASSERT(embeddingsScales.size() == (m - 1));

    const int warps_per_threadblock = 4;
    const int threadblock_size = warps_per_threadblock * 32;
    const int K_BLOCKS = (k + 32 - 1) / 32 * 2;
    const int M_BLOCKS = (m + 32 - 1) / 32 * 2;
    const int SKEW_HALF = 8;
    const int SMEM_STRIDE = (K_BLOCKS * 16 + SKEW_HALF);
    const int smem_rows_per_warp = (m > 16) ? ((m + 1) / 2 * 2) : 16; // multiple of 2 to guarantee 256-bit alignment for start of the row, at least 16 to safeload a tile
    const int smem_elems_per_warp_mat = smem_rows_per_warp * SMEM_STRIDE;
    const int SKEW_HALF_ACC = 8;
    const int SMEM_STRIDE_ACC = (M_BLOCKS * 16 + SKEW_HALF_ACC);
    const int smem_elems_per_warp_acc = M_BLOCKS * 16 * SMEM_STRIDE_ACC;
    const int smem_elems_per_warp = (smem_elems_per_warp_mat > smem_elems_per_warp_acc) ? smem_elems_per_warp_mat : smem_elems_per_warp_acc;
    const int LOADS_IN_FLIGHT = 13;
    const int STORES_IN_FLIGHT = 4;
    const bool remapIndices = indexRemapping != nullptr;
    EmbeddingScales scales;
    for(int i = 0; i < m - 1; ++i)
        scales.vals[i] = __float2half_rn(embeddingsScales[i]);

    runFp16Kernel5(remapIndices,inOutType,M_BLOCKS,K_BLOCKS);
    CUDA_ASSERT(cudaGetLastError());
    return 0;
}

void * allocateHelperDataForBatchedSyrkInt8(
    int outputInteractionsPadded,
    const std::vector<float>& embeddingsScales,
    float inScale,
    float outScale)
{
    std::vector<float> scales;
    scales.push_back(inScale / sqrtf(outScale));
    for(float scale: embeddingsScales)
        scales.push_back(scale / sqrtf(outScale));
    if (scales.size() % 2 == 1)
        scales.push_back(0.0F);

    void * helperData;
    CUDA_ASSERT(cudaMalloc(&helperData, scales.size() * sizeof(float)));
    CUDA_ASSERT(cudaMemcpy(helperData, &scales.front(), scales.size() * sizeof(float), cudaMemcpyHostToDevice));
    return helperData;
}

void deallocateHelperDataForBatchedSyrkInt8(void * helperData)
{
    CUDA_ASSERT(cudaFree(helperData));
}

#define runInt8Kernel(SAMPLES_PER_WARP_CONST,OUTPUT_INTERLEAVED_CONST,REMAP_INDICES_CONST,M_BLOCKS_CONST,K_BLOCKS_CONST) \
    batchedIsyrk<SAMPLES_PER_WARP_CONST,OUTPUT_INTERLEAVED_CONST,REMAP_INDICES_CONST,warps_per_threadblock,threadblock_size,M_BLOCKS_CONST,K_BLOCKS_CONST,(K_BLOCKS_CONST * 16 + SKEW_HALF),LOADS_IN_FLIGHT,LOADS_IN_FLIGHT_LDGSTS><<<(batchSize+samplesPerThreadblock-1)/samplesPerThreadblock,threadblock_size,warps_per_threadblock*smem_elems_per_warp,stream>>>( \
        (const int8_t *)denseInput, sparseInput, (const int8_t *)embeddings, (const int8_t *)embeddingsHost, indexRemapping, \
        tableOffsets, inScale / outScale, (const float *)helperData, (int8_t *)output, m, batchSize, embeddingRowsOnDevice, \
        outputSizePadded, smem_elems_per_sample, smem_rows_per_sample)
#define runInt8Kernel2(SAMPLES_PER_WARP_CONST,OUTPUT_INTERLEAVED_CONST,REMAP_INDICES,M_BLOCKS_CONST,K_BLOCKS_CONST) \
        if (REMAP_INDICES) \
        { \
            runInt8Kernel(SAMPLES_PER_WARP_CONST,OUTPUT_INTERLEAVED_CONST,true,M_BLOCKS_CONST,K_BLOCKS_CONST); \
        } \
        else \
        { \
            runInt8Kernel(SAMPLES_PER_WARP_CONST,OUTPUT_INTERLEAVED_CONST,false,M_BLOCKS_CONST,K_BLOCKS_CONST); \
        }
#define runInt8Kernel3(SAMPLES_PER_WARP_CONST,OUTPUT_INTERLEAVED,REMAP_INDICES,M_BLOCKS_CONST,K_BLOCKS_CONST) \
        if (OUTPUT_INTERLEAVED) \
        { \
            runInt8Kernel2(SAMPLES_PER_WARP_CONST,true,REMAP_INDICES,M_BLOCKS_CONST,K_BLOCKS_CONST); \
        } \
        else \
        { \
            runInt8Kernel2(SAMPLES_PER_WARP_CONST,false,REMAP_INDICES,M_BLOCKS_CONST,K_BLOCKS_CONST); \
        }
#define runInt8Kernel4(SAMPLES_PER_WARP,OUTPUT_INTERLEAVED,REMAP_INDICES,M_BLOCKS_CONST,K_BLOCKS_CONST) \
        switch (SAMPLES_PER_WARP) \
        { \
            case 1: \
                runInt8Kernel3(1,OUTPUT_INTERLEAVED,REMAP_INDICES,M_BLOCKS_CONST,K_BLOCKS_CONST); \
                break; \
            case 2: \
                runInt8Kernel3(2,OUTPUT_INTERLEAVED,REMAP_INDICES,M_BLOCKS_CONST,K_BLOCKS_CONST); \
                break; \
            default: \
                ASSERT(0); \
                break; \
        }
#define runInt8Kernel5(SAMPLES_PER_WARP,OUTPUT_INTERLEAVED,REMAP_INDICES,M_BLOCKS,K_BLOCKS_CONST) \
        switch (M_BLOCKS) \
        { \
            case 4: \
                runInt8Kernel4(SAMPLES_PER_WARP,OUTPUT_INTERLEAVED,REMAP_INDICES,4,K_BLOCKS_CONST); \
                break; \
            case 8: \
                runInt8Kernel4(SAMPLES_PER_WARP,OUTPUT_INTERLEAVED,REMAP_INDICES,8,K_BLOCKS_CONST); \
                break; \
            default: \
                ASSERT(0); \
                break; \
        }
#define runInt8Kernel6(SAMPLES_PER_WARP,OUTPUT_INTERLEAVED,REMAP_INDICES,M_BLOCKS,K_BLOCKS) \
        switch (K_BLOCKS) \
        { \
            case 2: \
                runInt8Kernel5(SAMPLES_PER_WARP,OUTPUT_INTERLEAVED,REMAP_INDICES,M_BLOCKS,2); \
                break; \
            case 4: \
                runInt8Kernel5(SAMPLES_PER_WARP,OUTPUT_INTERLEAVED,REMAP_INDICES,M_BLOCKS,4); \
                break; \
            case 6: \
                runInt8Kernel5(SAMPLES_PER_WARP,OUTPUT_INTERLEAVED,REMAP_INDICES,M_BLOCKS,6); \
                break; \
            case 8: \
                runInt8Kernel5(SAMPLES_PER_WARP,OUTPUT_INTERLEAVED,REMAP_INDICES,M_BLOCKS,8); \
                break; \
            default: \
                ASSERT(0); \
                break; \
        }
        
int runBatchedSyrkInt8(
    cudaStream_t stream,
    int m,
    int k,
    int batchSize,
    int embeddingRowsOnDevice,
    int outputSizePadded,
    const void * denseInput,
    const int * sparseInput,
    const void * embeddings,
    const void * embeddingsHost,
    const int * indexRemapping,
    const int * tableOffsets,
    const void * helperData,
    void * output,
    float inScale,
    float outScale,
    bool outputInterleaved)
{
    ASSERT((k % 32) == 0);
    ASSERT(k <= 128);
    ASSERT(m <= 64);

    const int warps_per_threadblock = 2;
    const int threadblock_size = warps_per_threadblock * 32;
    int deviceId;
    CUDA_ASSERT(cudaGetDevice(&deviceId));
    int smemSize;
    CUDA_ASSERT(cudaDeviceGetAttribute(&smemSize, cudaDevAttrMaxSharedMemoryPerMultiprocessor, deviceId));
    int samplesPerWarp = (smemSize >= 128 * 1024) ? 2 : 1;
    ASSERT(batchSize % samplesPerWarp == 0);
    const int K_BLOCKS = (k + 32 - 1) / 32 * 2;
    const int M_BLOCKS = (m + 32 - 1) / 32 * 4;
    const int SKEW_HALF = 16;
    const int SMEM_STRIDE = (K_BLOCKS * 16 + SKEW_HALF);
    const int smem_rows_per_sample = (m > 8) ? ((m + 1) / 2 * 2) : 8; // multiple of 2 to guarantee 256-bit alignment for start of the row, at least 8 to safeload a tile
    const int smem_elems_per_sample_mat = smem_rows_per_sample * SMEM_STRIDE;
    const int smem_elems_per_sample = smem_elems_per_sample_mat;
    const int smem_elems_per_warp = smem_elems_per_sample * samplesPerWarp;
    const int LOADS_IN_FLIGHT = 13;
    const int LOADS_IN_FLIGHT_LDGSTS = (LOADS_IN_FLIGHT + 1) / 2;
    const int samplesPerThreadblock = warps_per_threadblock * samplesPerWarp;
    const bool remapIndices = indexRemapping != nullptr;

    runInt8Kernel6(samplesPerWarp,outputInterleaved,remapIndices,M_BLOCKS,K_BLOCKS);
    CUDA_ASSERT(cudaGetLastError());
    return 0;
}

__global__ void remapEmbeddingRows(
    const int8_t * __restrict srcEmbeddings,
    int8_t * __restrict dstEmbeddings,
    const int * __restrict newLocations,
    int embeddingSize,
    int embeddingRows,
    int maxEmbeddingRowaGpu)
{
    int rowId = blockIdx.x;
    int elemId = threadIdx.x;
    int newRowPos = newLocations[rowId];
    int8_t val = srcEmbeddings[(long long)rowId * embeddingSize + elemId];
    if (newRowPos < maxEmbeddingRowaGpu)
        dstEmbeddings[(long long)newRowPos * embeddingSize + elemId] = val;
}

void remapEmbeddingRows(
    cudaStream_t stream,
    const int8_t * srcEmbeddings,
    int8_t * dstEmbeddings,
    const int * newLocations,
    int embeddingSize,
    int embeddingRows,
    int maxEmbeddingRowaGpu)
{
    remapEmbeddingRows<<<embeddingRows, embeddingSize, 0, stream>>>(
        srcEmbeddings,
        dstEmbeddings,
        newLocations,
        embeddingSize,
        embeddingRows,
        maxEmbeddingRowaGpu);
    CUDA_ASSERT(cudaGetLastError());
}