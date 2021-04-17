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

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

enum InOutDataType
{
    FLOAT,
    HALF,
    INT8
};

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
    const int * indicesOffsets,
    const std::vector<float>& embeddingsScales,
    void * output,
    InOutDataType inOutType);

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
    const int * indicesOffsets,
    const void * helperData,
    void * output,
    float inScale,
    float outScale,
    bool outputInterleaved);

void remapEmbeddingRows(
    cudaStream_t stream,
    const int8_t * srcEmbeddings,
    int8_t * dstEmbeddings,
    const int * newLocations,
    int embeddingSize,
    int embeddingRows,
    int maxEmbeddingRowaGpu);

void * allocateHelperDataForBatchedSyrkInt8(
    int outputInteractionsPadded,
    const std::vector<float>& embeddingsScales,
    float inScale,
    float outScale);

void deallocateHelperDataForBatchedSyrkInt8(void * helperData);
