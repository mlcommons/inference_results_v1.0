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

#include "cuda_fp16.h"

 
#define cublasErrCheck(stat) { cublasErrCheck_((stat), __FILE__, __LINE__); }
void cublasErrCheck_(cublasStatus_t stat, const char *file, int line)
{
    if (stat != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "cuBLAS Error: %d %s %d\n", stat, file, line);
    }
}

__host__ __device__ int roundoff(int v, int d) {
   return (v + d - 1) / d * d;
}
 
// Device functions
__forceinline__ __device__ float sigmoidf(float in) {
    return 1.f / (1.f + __expf(-in));  
}

#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line)
{
    if (stat != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
    }
}


__global__ void printTensor_ker(half *t, int n) {
    for (int i = 0; i < n; i++) {
        if ((float)t[i] !=  (float)t[i]) {           
            printf("%d %f\n", i, (float)t[i]);
            break;
        }
    }
    
}

void printTensor(half *t, int n) {
    printTensor_ker <<< 1, 1, 0, 0 >>> (t, n);
    
}


template<typename T_GEMM_IN, typename T_GEMM_OUT, typename T_BIAS, int blockSize>
__global__ void elementWise_fp(int hiddenSize, 
                               int batchSize,
                               T_GEMM_OUT *tmp_h, 
                               T_GEMM_OUT *tmp_i, 
                               T_BIAS *bias,
                               half *h_out,
                               T_GEMM_IN *i_out,
                               half *c_in,
                               half *c_out,
                               int layer,
                               int numLayers) {
    int numElements = batchSize * hiddenSize;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (index >= numElements) return;
    
    // TODO: fast divmod.
    int example = index / hiddenSize;
    int gateIndex = (index % hiddenSize) + 4 * example * hiddenSize;    
    
    float activationIn[4];
    
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        activationIn[i] = (float)tmp_h[i * hiddenSize + gateIndex] + (float)tmp_i[i * hiddenSize + gateIndex];
        
        
        activationIn[i] += (float)bias[i * hiddenSize + index % hiddenSize] + (float)bias[(i + 4) * hiddenSize + index % hiddenSize];
        // if (index == 0) printf("%d %f %f %f %f\n", i, (float)tmp_h[i * hiddenSize + gateIndex], (float)tmp_i[i * hiddenSize + gateIndex], (float)bias[i * hiddenSize + index % hiddenSize], (float)bias[(i + 4) * hiddenSize + index % hiddenSize]);
    }
    
    float in_gate      = sigmoidf(activationIn[0]);
    float forget_gate  = sigmoidf(activationIn[1]);
    float in_gate2     = tanhf(activationIn[2]);
    float out_gate     = sigmoidf(activationIn[3]);
    
    // if (index == 0) printf("%f %f %f %f\n", in_gate, forget_gate, in_gate2, out_gate);
    
    int hcIndex = index % hiddenSize + layer * hiddenSize + example * numLayers * hiddenSize;
    
    
    float val = (forget_gate * (float)c_in[hcIndex]) + (in_gate * in_gate2);
    
    c_out[hcIndex] = val;
    
    val = out_gate * tanhf(val);

    T_GEMM_IN outH;
    T_GEMM_IN outY;
    
    outH = val;
    h_out[hcIndex] = outH;
  
    outY = outH;
    i_out[index] = outY;
    
    // if (index == 0) printf("%f %f %f %f\n", (float)c_in[hcIndex], (float)c_out[hcIndex], (float)h_out[hcIndex], (float)i_out[index] );
}

template<typename T_GEMM_IN, cudaDataType_t dataTypeIn, typename T_GEMM_OUT, cudaDataType_t dataTypeOut, typename T_BIAS>
void decoderStepOld(int hiddenSize, 
              int inputSize,
              int batchSize, 
              int seqLength, 
              int numLayers,
              cublasHandle_t cublasHandle,
              T_GEMM_IN *x, 
              T_GEMM_IN *hx, 
              half *cx, 
              T_GEMM_IN **w, 
              T_BIAS **bias,
              half *y, 
              half *hy, 
              half *cy,
              T_GEMM_IN *tmp_io,
              T_GEMM_OUT *tmp_i,
              T_GEMM_OUT *tmp_h,
              cudaStream_t streami,
              cudaStream_t streamh) {
    T_GEMM_OUT alphaR = 1.f;
    T_GEMM_OUT betaR  = 0.f;    
    
    T_GEMM_OUT alphaL = 1.f;
    T_GEMM_OUT betaL  = 0.f;       

    int numElements = hiddenSize * batchSize;
    
    const cublasOperation_t transa = CUBLAS_OP_T;
    const cublasOperation_t transb = CUBLAS_OP_N;
    
    if (seqLength > 1) {
        printf("Seq length > 1 not supported in this code.\n");
        return;
    }
    
    cudaEvent_t streamh_event;
    cudaEvent_t streami_event;
    cudaErrCheck(cudaEventCreate(&streamh_event, cudaEventDisableTiming));
    cudaErrCheck(cudaEventCreate(&streami_event, cudaEventDisableTiming));
    
    for (int layer = 0; layer < numLayers; layer++) {
        T_GEMM_IN *layer_i_in = layer == 0 ? x : tmp_io + numElements * (layer - 1);
        T_GEMM_IN *layer_i_out = layer == numLayers - 1 ? (T_GEMM_IN*)y : tmp_io + numElements * layer;
       

        // Run these in parallel with each other
        
        T_GEMM_IN *inData = layer_i_in;
        T_GEMM_IN *wData = w[layer];        
        
        cublasErrCheck(cublasSetStream(cublasHandle, streami));
        
        cublasErrCheck(cublasGemmEx(cublasHandle,
                                    transa, transb,
                                    4 * hiddenSize, batchSize, (layer == 0 ? inputSize : hiddenSize),
                                    &alphaL,
                                    wData,
                                    dataTypeIn,
                                    transa == CUBLAS_OP_N ? 4 * hiddenSize : (layer == 0 ? inputSize : hiddenSize),
                                    inData,
                                    dataTypeIn,
                                    (layer == 0 ? inputSize : hiddenSize),
                                    &betaL,
                                    tmp_i,
                                    dataTypeOut,
                                    4 * hiddenSize,
                                    dataTypeOut,
                                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        cudaErrCheck(cudaEventRecord(streami_event, streami));  
        
        // Run recurrent GEMM in stream H (still in parallel with GEMMs above)
        cublasErrCheck(cublasSetStream(cublasHandle, streamh));

        cublasErrCheck(cublasGemmEx(cublasHandle,
                                    transa, transb,
                                    4 * hiddenSize, batchSize, hiddenSize,
                                    &alphaR,
                                    wData + 4 * hiddenSize * (layer == 0 ? inputSize : hiddenSize), 
                                    dataTypeIn,
                                    transa == CUBLAS_OP_N ? 4 * hiddenSize : hiddenSize,
                                    // hx + layer * hiddenSize * batchSize,
                                    hx + layer * hiddenSize,
                                    dataTypeIn,
                                    // hiddenSize,
                                    numLayers * hiddenSize,
                                    &betaR,
                                    // tmp_h + 4 * layer * numElements, 
                                    tmp_h, 
                                    dataTypeOut,
                                    4 * hiddenSize,
                                    dataTypeOut,
                                    CUBLAS_GEMM_DEFAULT_TENSOR_OP)); 

        cudaErrCheck(cudaStreamWaitEvent(streamh, streami_event, 0));

        dim3 blockDim;
        dim3 gridDim;
        
        blockDim.x = 256;
        gridDim.x = (numElements + blockDim.x - 1) / blockDim.x;
        
        elementWise_fp<T_GEMM_IN, T_GEMM_OUT, T_BIAS, 256> <<< gridDim, blockDim , 0, streamh >>> 
                 (hiddenSize, 
                  batchSize,
                  // tmp_h + 4 * layer * numElements, 
                  tmp_h, 
                  tmp_i,
                  bias[layer],
                  // hy + layer * hiddenSize * batchSize,
                  hy,
                  layer_i_out,
                  // cx + layer * hiddenSize * batchSize,
                  // cy + layer * hiddenSize * batchSize,
                  cx,
                  cy,
                  layer,
                  numLayers);

        cudaErrCheck(cudaGetLastError());
        
        // h stream need to wait for eltwise op to complete 
        cudaErrCheck(cudaEventRecord(streamh_event, streamh));          
        cudaErrCheck(cudaStreamWaitEvent(streami, streamh_event, 0));
        
    }
    cudaErrCheck(cudaEventDestroy(streami_event));  
    cudaErrCheck(cudaEventDestroy(streamh_event));  
}



template<typename T_GEMM_IN, cudaDataType_t dataTypeIn, typename T_GEMM_OUT, cudaDataType_t dataTypeOut, typename T_BIAS>
void decoderStep(int hiddenSize, 
              int inputSize,
              int batchSize, 
              int seqLength, 
              int numLayers,
              cublasHandle_t cublasHandle,
              T_GEMM_IN *x, 
              T_GEMM_IN *hx, 
              half *cx, 
              T_GEMM_IN **w, 
              T_BIAS **bias,
              half *y, 
              half *hy, 
              half *cy,
              T_GEMM_IN *tmp_io,
              T_GEMM_OUT *tmp_i,
              T_GEMM_OUT *tmp_h,
              cudaStream_t streami,
              cudaStream_t streamh) {
    T_GEMM_OUT alphaR = 1.f;
    T_GEMM_OUT betaR  = 0.f;    
    
    T_GEMM_OUT alphaL = 1.f;
    T_GEMM_OUT betaL  = 0.f;       

    int numElements = hiddenSize * batchSize;
    
    const cublasOperation_t transa = CUBLAS_OP_T;
    const cublasOperation_t transb = CUBLAS_OP_N;
    
    if (seqLength > 1) {
        printf("Seq length > 1 not supported in this code.\n");
        return;
    }
    
    cudaEvent_t streamh_event;
    cudaEvent_t streami_event;
    cudaErrCheck(cudaEventCreate(&streamh_event, cudaEventDisableTiming));
    cudaErrCheck(cudaEventCreate(&streami_event, cudaEventDisableTiming));
    
    for (int layer = 0; layer < numLayers; layer++) {
        T_GEMM_IN *layer_i_in = layer == 0 ? x : tmp_io + numElements * (layer - 1);
        T_GEMM_IN *layer_i_out = layer == numLayers - 1 ? (T_GEMM_IN*)y : tmp_io + numElements * layer;

        // Run these in parallel with each other
        
        T_GEMM_IN *inData = layer_i_in;
        T_GEMM_IN *wData = w[layer];        

        // Run recurrent GEMM in stream H (still in parallel with GEMMs above)
        cublasErrCheck(cublasSetStream(cublasHandle, streamh));

        cublasErrCheck(cublasGemmEx(cublasHandle,
                                    transa, transb,
                                    4 * hiddenSize, batchSize, hiddenSize,
                                    &alphaR,
                                    wData + 4 * hiddenSize * (layer == 0 ? inputSize : hiddenSize), 
                                    dataTypeIn,
                                    transa == CUBLAS_OP_N ? 4 * hiddenSize : hiddenSize,
                                    hx + layer * hiddenSize,
                                    dataTypeIn,
                                    numLayers * hiddenSize,
                                    &betaR,
                                    tmp_h + 4 * layer * numElements, 
                                    dataTypeOut,
                                    4 * hiddenSize,
                                    dataTypeOut,
                                    CUBLAS_GEMM_DEFAULT_TENSOR_OP)); 
  
        cudaErrCheck(cudaEventRecord(streamh_event, streamh));  

        // Run input GEMM in stream I (still in parallel with GEMMs above)
        cublasErrCheck(cublasSetStream(cublasHandle, streami));
        
        cublasErrCheck(cublasGemmEx(cublasHandle,
                                    transa, transb,
                                    4 * hiddenSize, batchSize, (layer == 0 ? inputSize : hiddenSize),
                                    &alphaL,
                                    wData,
                                    dataTypeIn,
                                    transa == CUBLAS_OP_N ? 4 * hiddenSize : (layer == 0 ? inputSize : hiddenSize),
                                    inData,
                                    dataTypeIn,
                                    (layer == 0 ? inputSize : hiddenSize),
                                    &betaL,
                                    tmp_i,
                                    dataTypeOut,
                                    4 * hiddenSize,
                                    dataTypeOut,
                                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));


        cudaErrCheck(cudaStreamWaitEvent(streami, streamh_event, 0));

        dim3 blockDim;
        dim3 gridDim;
        
        blockDim.x = 256;
        gridDim.x = (numElements + blockDim.x - 1) / blockDim.x;
        
        elementWise_fp<T_GEMM_IN, T_GEMM_OUT, T_BIAS, 256> <<< gridDim, blockDim , 0, streami >>> 
                 (hiddenSize, 
                  batchSize,
                  tmp_h + 4 * layer * numElements, 
                  tmp_i,
                  bias[layer],
                  hy,
                  layer_i_out,
                  cx,
                  cy,
                  layer,
                  numLayers);

        cudaErrCheck(cudaGetLastError());
    }
    cudaErrCheck(cudaEventDestroy(streami_event));  
    cudaErrCheck(cudaEventDestroy(streamh_event));  
}


