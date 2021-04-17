/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
#include <stdint.h>

void greedySearch(int32_t* symbolArr,
                  int32_t* num_symbols_current_step,
                  int32_t* encIdx,
                  int32_t* seqLen,
                  bool* isNotBlank,
                  int32_t* outSeq,
                  int32_t* outSeqLen,
                  int32_t* done,
                  int batchSize,
                  int iter,
                  int _BLANK_,
                  int hp_max_symbols_per_step,
                  int maxLength,
                  cudaStream_t stream);


void rnntSparseMemSet(uint8_t* devBuffer,
                      bool* sparseMask,
                      int sizeBytes,
                      int batchSize,
                      cudaStream_t stream);

void rnntFc2Top1(uint8_t* devFc1EncoderBuffer,
                 uint8_t* devFc1DecoderBuffer,
                 uint8_t* devFc1WeightsBuffer,
                 uint8_t* devFc1BiasBuffer,
                 int32_t* devFc2OutputBuffer,
                 int32_t* devTop1Buffer,
                 int batchSize,
                 cudaStream_t stream);


void rnntIgatherStep(
                 uint8_t* devRnntEncoderBuffer,
                 int32_t* devEncIdxBuffer,
                 uint8_t* devEncGatherBuffer,
                 size_t  eSize,
                 size_t  encoderChannels,
                 size_t  seqLength,
                 int batchSize,
                 cudaStream_t stream);

