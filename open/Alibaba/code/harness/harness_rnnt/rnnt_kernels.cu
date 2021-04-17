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

#include <stdio.h>
#include <cuda_fp16.h>
#include <stdint.h> // uint8_t
#include <cassert>

// ====================
// Greedy search
// ====================
//     Implementing vanilla greedy search in the device

__global__ void greedySearch_ker(int32_t* symbolArr,
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
                                 int maxLength) {
    int bs = blockIdx.x * blockDim.x + threadIdx.x;

    if (bs >= batchSize) return;

    /* for (size_t bs = 0; bs < actualBatchSize ; bs++)
    {
        // Get current winner symbol
        int32_t winner_symbol   = symbolArr[bs];

        // Update state based on the outcome
        if(winner_symbol != _BLANK_ && num_symbols_current_step[bs] < FLAGS_hp_max_symbols_per_step)
        {
            isNotBlank[bs] = true;
            lastSymbol[bs] = winner_symbol;

            // Note that here we do not update the time pointer because
            // we want to generate more symbols from the predictor

            // FLAGS_always_advance_time: update time pointers as a hack if we do not get real data
            if(FLAGS_always_advance_time)
            {
                if(encIdx[bs] < seqLen[bs])
                {
                    encIdx[bs]++;
                    num_symbols_current_step[bs] = 0;
                }
            }

            // Update output
            outSeq[bs].push(winner_symbol);
            num_symbols_current_step[bs]++;

            // std::cout << "[blank=false] BS : " << bs << " t=" << encIdx[bs] << " winner symbol:[" << winner_symbol << "]" << std::endl;
            // outSeq[bs].print("output");
        }
        else  // winner_symbol == _BLANK_
        {
            isNotBlank[bs] = false;

            // Note that here we do not update the inputs to the predictor
            // because we will do the prediction again (brute force) and we
            // want the same outcome

            // update time pointer
            if(encIdx[bs] < seqLen[bs])
            {
                encIdx[bs]++;
                num_symbols_current_step[bs] = 0;
            }
        }

        if(encIdx[bs] < seqLen[bs]) done = false;
    } */

    // Get current winner symbol
    int32_t winner_symbol   = symbolArr[bs];

    // Update state based on the outcome
    // if(winner_symbol != _BLANK_ && num_symbols_current_step[bs] < hp_max_symbols_per_step)
    if(winner_symbol != _BLANK_ && num_symbols_current_step[bs] < hp_max_symbols_per_step && (encIdx[bs] < seqLen[bs]))
    {
        isNotBlank[bs] = true;

        // Note that here we do not update the time pointer because
        // we want to generate more symbols from the predict

        // Update output
        outSeq[bs * maxLength + outSeqLen[bs]] = winner_symbol; // TODO: Fix bad access pattern
        outSeqLen[bs]++;
        num_symbols_current_step[bs]++;

        // std::cout << "[blank=false] BS : " << bs << " t=" << encIdx[bs] << " winner symbol:[" << winner_symbol << "]" << std::endl;
        // outSeq[bs].print("output");
    }
    else  // winner_symbol == _BLANK_
    {
        isNotBlank[bs] = false;

        // Note that here we do not update the inputs to the predictor
        // because we will do the prediction again (brute force) and we
        // want the same outcome

        // update time pointer
        if(encIdx[bs] < seqLen[bs]) {
            encIdx[bs]++;
            num_symbols_current_step[bs] = 0;
        }
    }

    // check if these batch instance is done (consumer the whole encoder input)
    // TODO: Hammering *done from many threads in a block. Maybe do a _any across the warp/CTA.
    if(encIdx[bs] < seqLen[bs]) {
        // printf("not done\n");
        done[iter] = 0;
    }
}


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
                  cudaStream_t stream) {

    dim3 blockDim = dim3(128, 1, 1);
    dim3 gridDim = dim3((batchSize + blockDim.x - 1) / blockDim.x, 1, 1);


    greedySearch_ker <<< gridDim, blockDim, 0, stream >>> (symbolArr,
                                                           num_symbols_current_step,
                                                           encIdx,
                                                           seqLen,
                                                           isNotBlank,
                                                           outSeq,
                                                           outSeqLen,
                                                           done,
                                                           batchSize,
                                                           iter,
                                                           _BLANK_,
                                                           hp_max_symbols_per_step,
                                                           maxLength);

}





// ====================
// Sparse Initializers
// ====================
//     A family of methods to implement conditional/sparse memsets to state/cell tensors in the LSTMs of the models


//     void InitializeEncoderSparse(bool *sparseMask, size_t actualBatchSize)
//     {
//         size_t encPreSize  = FLAGS_hp_enc_pre_rnn_layers * FLAGS_hp_encoder_hidden_size * esize;
//         size_t encPostSize = FLAGS_hp_enc_post_rnn_layers * FLAGS_hp_encoder_hidden_size * esize;
//
//         int8_t* pEncoderPreHidden  = (int8_t*) encoderPreHidden->data();
//         int8_t* pEncoderPreCell    = (int8_t*) encoderPreCell->data();
//         int8_t* pEncoderPostHidden = (int8_t*) encoderPostHidden->data();
//         int8_t* pEncoderPostCell   = (int8_t*) encoderPostCell->data();
//
//         // Apply initialization element by element (preserve content when sparseMask is true)
//         for (size_t bs = 0 ; bs < actualBatchSize ; bs++)
//         {
//             if(sparseMask[bs] == false) {
//                 CHECK_EQ(cudaMemset(pEncoderPreHidden,  0, encPreSize),  cudaSuccess);
//                 CHECK_EQ(cudaMemset(pEncoderPreCell,    0, encPreSize),  cudaSuccess);
//                 CHECK_EQ(cudaMemset(pEncoderPostHidden, 0, encPostSize), cudaSuccess);
//                 CHECK_EQ(cudaMemset(pEncoderPostCell,   0, encPostSize), cudaSuccess);
//             }
//
//             // update pointers
//             pEncoderPreHidden  += encPreSize;
//             pEncoderPreCell    += encPreSize;
//             pEncoderPostHidden += encPostSize;
//             pEncoderPostCell   += encPostSize;
//         }
//     }


// Given tensors of size  batchSize*int32
//   [0] ptr = devBuffer + bs * int32
//   [1] Set to zero is sparseMask[i] == true
//   [2] Update devBuffer pointer with stride, got to step [1]

__global__ void rnnt_sparse_memset_ker(int32_t* buffer,
                                 bool* sparseMask,
                                 int iter,
                                 int stride,
                                 int numThreads)
{
    int thr = blockIdx.x * blockDim.x + threadIdx.x;
    if (thr >= numThreads) return;

    int i;
    int32_t  index = thr;

    for(i=0; i<iter; i++) {
       if(sparseMask[i] == false) {
          buffer[index] = 0;
       }
       index += stride;
    }
}

void rnntSparseMemSet(uint8_t* devBuffer,
                      bool* sparseMask,
                      int sizeBytes,
                      int batchSize,
                      cudaStream_t stream) {

    int32_t *buffer32 = (int32_t*)devBuffer;
    int iter = batchSize;
    int numThreads = sizeBytes / sizeof(int32_t);
    int32_t stride = sizeBytes / sizeof(int32_t);
    dim3 blockDim = dim3(128, 1, 1);
    dim3 gridDim  = dim3((numThreads + blockDim.x - 1) / blockDim.x, 1, 1);

    // printf("rnntSparseMemSet(size=%d,bs=%d): iter=%d numThreads=%d \n",sizeBytes, batchSize, iter, numThreads);
    rnnt_sparse_memset_ker <<< gridDim, blockDim, 0, stream >>> (buffer32,
                                                                 sparseMask,
                                                                 iter,
                                                                 stride,
                                                                 numThreads);
}

// ====================
// Joint FC1+top1
// ====================
//     Implementing  FC1_SUM + RELU + FC2 + TOP1 into a single cuda kernel


#if 1

template <int K, int NUM_WARPS, int ROWS_PER_WARP, int COLS_PER_BLOCK, bool save_intermediate>
__launch_bounds__(NUM_WARPS*32,2)
__global__ void fc2_top1_ker(half2* A, half2* B1, half2*B2, half* C, half* d_bias, int32_t* top1) {

    const int K2 = K/2;

    half2 accum[COLS_PER_BLOCK][ROWS_PER_WARP];

    int warp_id = threadIdx.x / 32;
    int tid = threadIdx.x % 32;

    int warp_row = warp_id * ROWS_PER_WARP;

    // Delay some warps in order to prevent overwhelming LSU
    /*
    uint64_t time = clock64();
    while (clock64() - time < warp_id*1200);
    */

    int sample_id = blockIdx.x*COLS_PER_BLOCK;

    for (int r=0; r<ROWS_PER_WARP; r++) {
        for (int ni=0; ni<COLS_PER_BLOCK; ni++) {
            accum[ni][r].x = 0;
            accum[ni][r].y = 0;
        }
    }

    bool pred[ROWS_PER_WARP];
    for (int r=0; r<ROWS_PER_WARP; r++) {
        pred[r] = warp_row + r < 29;
    }

    half2 a[ROWS_PER_WARP][K2/32];
    half2 b[COLS_PER_BLOCK][K2/32];
    half bias[ROWS_PER_WARP];

#pragma unroll
    for (int i=0; i<K2/32; i++) {
#pragma unroll
        for (int mi=0; mi<ROWS_PER_WARP; mi++) {
            int row = warp_row + mi;
            if (pred[mi]) a[mi][i] = A[row*K2 + i*32 + tid];
        }
#pragma unroll
        for (int ni=0; ni<COLS_PER_BLOCK; ni++) {
            // apply here element_wise: RELU(SUM(B1, B2))
            // b[ni][i] = B[(sample_id+ni)*K2+i*32+tid];
            b[ni][i] = B1[(sample_id+ni)*K2+i*32+tid] + B2[(sample_id+ni)*K2+i*32+tid];
            // RELU: if (b[ni][i] < 0.0) b[ni][i] = 0.0;
            b[ni][i] = b[ni][i] * __hgeu2(b[ni][i], __float2half2_rn(0.0) );
        }
    }

    for (int mi=0; mi<ROWS_PER_WARP; mi++) {
        int row = warp_row + mi;
        if (pred[mi]) bias[mi] = d_bias[row];
    }

#pragma unroll
    for (int mi=0; mi<ROWS_PER_WARP; mi++) {
#pragma unroll
        for (int ni=0; ni<COLS_PER_BLOCK; ni++) {
#pragma unroll
            for (int i=0; i<K2/32; i++) {
                accum[ni][mi] += a[mi][i] * b[ni][i];
            }
        }
    }

    __shared__ float result[COLS_PER_BLOCK][32];

#pragma unroll
    for (int r=0; r<ROWS_PER_WARP; r++) {
#pragma unroll
        for (int ni=0; ni<COLS_PER_BLOCK; ni++) {
            // Warp reduce
            for (int offset=16; offset>0; offset /= 2) {
                accum[ni][r] += __shfl_down_sync(0xFFFFFFFF,accum[ni][r],offset);
            }
            half val = accum[ni][r].x + accum[ni][r].y;
            val += bias[r];
            // printf("[%f]",__half2float(val)); // DEBUG
            if (save_intermediate && tid == 0 && (warp_row+r<29)) C[(sample_id+ni)*29+warp_row+r] = val;
            if (tid == 0) result[ni][warp_id*ROWS_PER_WARP + r] = __half2float(val);
        }
    }
    __syncthreads();
    if (warp_id == 0) {
        for (int ni=0; ni<COLS_PER_BLOCK; ni++) {
            float val = result[ni][threadIdx.x];
            int idx = (threadIdx.x<29) ? (int)threadIdx.x : -1;
            for (int offset=16; offset>0; offset /= 2) {
                int other_idx = __shfl_down_sync(0xFFFFFFFF,idx,offset);
                float other_val = __shfl_down_sync(0xFFFFFFFF,val,offset);
                if (idx == -1 || (other_idx != -1 && other_val > val)) {
                    idx = other_idx;
                    val = other_val;
                }
            }
            if (threadIdx.x == 0) {
                top1[sample_id+ni] = idx;
                // printf("{%d,%d}:: [sample_id=%d][ni=%d]: %d\n",blockIdx.x, threadIdx.x, sample_id,ni,idx); // DEBUG
            }
        }
    }
}


// void rnntFc2Top1(half2* devFc1EncoderBuffer,  // activation #1 (from fc1 encoder)
//                  half2* devFc1DecoderBuffer,  // activation #2 (from fc1 decoder)
//                  half2* devFc1WeightsBuffer,  // FC2 weights
//                  half* devFc1BiasBuffer,     // FC2 bias
//                  half* devFc2OutputBuffer,   // transient FC2 output
//                  int32_t* devTop1Buffer,     // Top1 output
//                  int batchSize,
//                  cudaStream_t stream) {

void rnntFc2Top1(uint8_t* devFc1EncoderBuffer,  // activation #1 (from fc1 encoder)
                 uint8_t* devFc1DecoderBuffer,  // activation #2 (from fc1 decoder)
                 uint8_t* devFc1WeightsBuffer,  // FC2 weights
                 uint8_t* devFc1BiasBuffer,     // FC2 bias
                 int32_t* devFc2OutputBuffer,   // transient FC2 output
                 int32_t* devTop1Buffer,        // Top1 output
                 int batchSize,
                 cudaStream_t stream) {

    // static parameters
    // const int K = 1024;
    const int K = 512;    // fc2 input size (i.e., fc1 output size)
    const int M = 32;
    const int NUM_WARPS = 4;
    const int ROWS_PER_WARP = M/NUM_WARPS;
    const int COLS_PER_BLOCK = 1;
    const bool save_intermediate = false;

    // process parameters
    int num_blocks = batchSize / COLS_PER_BLOCK;

    // cuda kernel
    // printf("fc2_top1_kernel<K=%d, NUM_WARPS=%d, ROWS_PER_WARP=%d, COLS_PER_BLOCK=%d><<<%d,%d>>>",K, NUM_WARPS, ROWS_PER_WARP, COLS_PER_BLOCK, num_blocks, NUM_WARPS*32);

    fc2_top1_ker<K, NUM_WARPS, ROWS_PER_WARP, COLS_PER_BLOCK, save_intermediate> <<<num_blocks,NUM_WARPS*32, 0, stream>>> (
        (half2*) devFc1WeightsBuffer,
        (half2*) devFc1EncoderBuffer,
        (half2*) devFc1DecoderBuffer,
        (half*) devFc2OutputBuffer,
        (half*) devFc1BiasBuffer,
        (int32_t*) devTop1Buffer);
}


#endif



// ====================
// Encoder gather
// ====================
//     Implementing  RnntGather for the FC1 encoder input in joint network
//
//         ec.igather->step(actualBatchSize, mainStream,
//             tc.encoderOut[pol],
//             tc.encIdx,
//             tc.encGather);


__global__ void rnnt_enc_gather_ker(int32_t* outBuffer,  //  [bs][seq_len][inner_stride]
                                    int32_t* idxVec,     //  [bs]
                                    int32_t* inBuffer,   //  [bs][inner_stride]
                                    int32_t inner_iter,
                                    int32_t inner_stride,
                                    int32_t outer_iter,
                                    int32_t outer_stride,
                                    size_t  max_seq_length,
                                    int batchSize)
{

    int inner_offset = blockDim.x;
    int bs = blockIdx.x * outer_iter;

    // outer loop ([bs])
    for(int i = 0; i < outer_iter ; i++, bs++)
    {
        if(bs >= batchSize) continue;

        // channel is based on thread Idx
        int ch = threadIdx.x;

        // get index for the input
        int32_t idx = idxVec[bs];

        // base pointers
        int32_t *pIn  = inBuffer  + (bs * outer_stride) + (idx * inner_stride);
        int32_t *pOut = outBuffer + (bs * inner_stride);

        // inner loop ([ch])
        for(int j = 0; j < inner_iter ; j++, ch+=inner_offset)
        {
            if(ch >= inner_stride) continue;

            // return zero is out of bounds
            if (idx >= max_seq_length) {
                pOut[ch] = 0;
                continue;
            }

            // transfer
            pOut[ch] = pIn[ch];
        }
    }
}

void rnntIgatherStep(
                 uint8_t* devRnntEncoderBuffer, // input from encoder RNNT  [bs][seq_len][chan]
                 int32_t* devEncIdxBuffer,      // vector of indexes  [bs], int32
                 uint8_t* devEncGatherBuffer,   // output to be consumed by joint::fc1   [bs][chan]
                 size_t  eSize,                // element size
                 size_t  encoderChannels,      // channel size (hp_encoder_hidden_size)
                 size_t  seqLength,            // sequence length (hp_max_seq_length)
                 int batchSize,
                 cudaStream_t stream) {

    // tiling approach: (example)
    //     block dimension = 128
    //     inner iter      = encoder channel size / 128    (e.g. 512*2/4 over 128 = 2)
    //     grid  dimension = 128
    //     outer iter      = (batchSize / grid_dimension)'

    // static parameters
    const int BLOCK_DIM = 128;
    const int GRID_DIM  = 2048;

    // inner iterations
    int blockSize      = encoderChannels * eSize;
    int blockElements  = blockSize / sizeof(int32_t);
    // assert(blockElements > BLOCK_DIM);
    int inner_iter = blockElements / BLOCK_DIM;

    // outer iterations
    int grid_elements = (batchSize > GRID_DIM)? GRID_DIM : batchSize;
    int outer_iter    = (batchSize + grid_elements -1) / grid_elements;

    // dim3 gridDim  = dim3((numThreads + blockDim.x - 1) / blockDim.x, 1, 1);
    dim3 blockDim = dim3(BLOCK_DIM, 1, 1);
    dim3 gridDim  = dim3(grid_elements, 1, 1);

    // strides
    int32_t  inner_stride = blockElements;
    int32_t  outer_stride = seqLength * inner_stride;

    // printf("rnntIgatherStep(bs=%d, esize=%d,encChan=%d,seqLen=%d)<%d,%d>: inner_iter=%d inner_stride=%d outer_iter=%d outer_stride=%d\n ", batchSize, eSize, encoderChannels, seqLength, BLOCK_DIM, grid_elements, inner_iter, inner_stride, outer_iter, outer_stride);
    rnnt_enc_gather_ker <<< gridDim, blockDim, 0, stream >>> ((int32_t*) devEncGatherBuffer,
                                                              (int32_t*) devEncIdxBuffer,
                                                              (int32_t*) devRnntEncoderBuffer,
                                                              inner_iter,
                                                              inner_stride,
                                                              outer_iter,
                                                              outer_stride,
                                                              seqLength,
                                                              batchSize);

}



