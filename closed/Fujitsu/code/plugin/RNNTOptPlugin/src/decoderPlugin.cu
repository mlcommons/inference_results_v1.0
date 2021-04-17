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

#include <cuda.h>

#include "decoderPlugin.h"

#define CHECK(status)                                          \
    do                                                         \
    {                                                          \
        auto ret = (status);                                   \
        if (ret != 0)                                          \
        {                                                      \
            std::cout << "Cuda failure: " << ret << std::endl; \
            abort();                                           \
        }                                                      \
    } while (0)




using namespace nvinfer1;
using nvinfer1::plugin::RNNTDecoderPlugin;
using nvinfer1::plugin::RNNTDecoderPluginCreator;

REGISTER_TENSORRT_PLUGIN(RNNTDecoderPluginCreator);

RNNTDecoderPlugin::RNNTDecoderPlugin(const PluginFieldCollection *fc) {
    int idx = 0;
    
    mNumLayers = *(int*)(fc->fields[idx].data);
    idx++;
    
    mHiddenSize = *(int*)(fc->fields[idx].data);
    idx++;
    
    mInputSize = *(int*)(fc->fields[idx].data);
    idx++;
    
    mDataType = *(nvinfer1::DataType*)(fc->fields[idx].data);
    idx++;
    
    mWeights_h = (void**)malloc(mNumLayers * sizeof(void*));
    
    for (int i = 0; i < mNumLayers; i++) {        
        mWeights_h[i] = (void*)fc->fields[idx].data;
        idx++;
    }
    
    mBias_h = (void**)malloc(mNumLayers * sizeof(void*));
    for (int i = 0; i < mNumLayers; i++) {        
        mBias_h[i] = (void*)fc->fields[idx].data;
        idx++;
    }
}

RNNTDecoderPlugin::RNNTDecoderPlugin(const void* data, size_t length) {
    const char *d = static_cast<const char*>(data);
    // Use maybe_unused attribute when updating to CUDA_STANDARD C++17
    #ifndef NDEBUG
    auto d_start = d;
    #endif
    read<int>(d, mNumLayers);
    read<int>(d, mHiddenSize);
    read<int>(d, mInputSize);
    
    read<nvinfer1::DataType>(d, mDataType);
    
    mWeights_h = (void**)malloc(mNumLayers * sizeof(void*));
    for (int i = 0; i < mNumLayers; i++) {        
        size_t dataTypeSize = 0;
        dataTypeSize = sizeof(half);
        
        size_t sz = 4 * mHiddenSize * ((i == 0 ? mInputSize : mHiddenSize) + mHiddenSize) * dataTypeSize;
               

        mWeights_h[i] = malloc(sz);
        memcpy(mWeights_h[i], d, sz);
        d += sz;
    }
    
    mBias_h = (void**)malloc(mNumLayers * sizeof(void*));
    for (int i = 0; i < mNumLayers; i++) {        
        size_t dataTypeSize = 0;
        dataTypeSize = sizeof(half);
        
        size_t sz = 8 * mHiddenSize * dataTypeSize;

        mBias_h[i] = malloc(sz);
        memcpy(mBias_h[i], d, sz);
        d += sz;
    }
    
    assert(d == d_start + length);
}

const char* RNNTDecoderPlugin::getPluginType() const {
    return "RNNTDecoderPlugin";
}

const char* RNNTDecoderPlugin::getPluginVersion() const {
    return "1";
}

void RNNTDecoderPlugin::setPluginNamespace(const char* libNamespace) {
    mNamespace = libNamespace;
}

const char* RNNTDecoderPlugin::getPluginNamespace() const {
    return mNamespace.c_str();
}

void RNNTDecoderPlugin::destroy() {
    if (mWeights_h) {
        free(mWeights_h);
        mWeights_h = nullptr;
    }
    if (mBias_h) {
        free(mBias_h);
        mBias_h = nullptr;
    }
    delete this;
}

void RNNTDecoderPlugin::setCUDAInfo(cudaStream_t mStreamh, cublasHandle_t mCublas, void **mWeights_d, void **mBias_d, void *mWorkSpace_d) {
    this->mStreamh = mStreamh;
    this->mCublas = mCublas;
    this->mWeights_d = mWeights_d;
    this->mBias_d = mBias_d;
    this->mWorkSpace_d = mWorkSpace_d;
}

IPluginV2DynamicExt * RNNTDecoderPlugin::clone() const {    
    size_t sz = getSerializationSize();
    
    char *buff = (char*)malloc(getSerializationSize());
    
    serialize(buff);
   
    RNNTDecoderPlugin* ret = new RNNTDecoderPlugin(buff, sz);
    
    ret->setCUDAInfo(mStreamh, mCublas, mWeights_d, mBias_d, mWorkSpace_d);
    
    free(buff);
    
    return ret;
}

int RNNTDecoderPlugin::getNbOutputs() const {
    return 3;
}

// TODO: No idea if this needs batch size. Actually, don't know what's expected at all.
/* Dims RNNTDecoderPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) {
    assert(index >= 0 && index < this->getNbOutputs());
    
    if (index == 0) {
        return Dims3(inputs[0].d[0], mNumLayers, mHiddenSize);
    }
    else {        
        return Dims3(inputs[0].d[0], 1, mHiddenSize);
    }
} */

DimsExprs RNNTDecoderPlugin::getOutputDimensions (int outputIndex, const DimsExprs *inputs, int nbInputs, IExprBuilder &exprBuilder) {
    
    assert(outputIndex >= 0 && outputIndex < this->getNbOutputs());
    
    return inputs[outputIndex];
}

bool RNNTDecoderPlugin::supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) { 
    if (inOut[pos].format != TensorFormat::kNCHW)
        return false;

    // fp16 I/O
    if (mDataType == nvinfer1::DataType::kHALF) {
        bool allHalf = true;

        // Don't care about pos. If all are half pass it.
        // The way this is called doesn't fill all of inOut, it only fills it up to pos.
        for (int i = 0; i <= pos; i++) {
            if (inOut[i].type != DataType::kHALF) {
                allHalf = false;
            }
        }
        
        if (allHalf) {
            return true;
        }
        return false;
    }
    return false;
}

/* void RNNTDecoderPlugin::configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput) {
    mInputSize = in[0].dims.d[in[0].dims.nbDims - 1];
} */

void RNNTDecoderPlugin::configurePlugin (const DynamicPluginTensorDesc *in, int nbInputs, const DynamicPluginTensorDesc *out, int nbOutputs) {
    // mInputSize = in[0].desc.dims.d[in[0].desc.dims.nbDims - 1];
}

// void RNNTDecoderPlugin::configurePlugin(const Dims *inputDims, int nbInputs, const Dims *outputDims, int nbOutputs, const DataType *inputTypes, const DataType *outputTypes, const bool *inputIsBroadcast, const bool *outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) {
    // mInputSize = inputDims[0].d[inputDims[0].nbDims - 1];
// }



int RNNTDecoderPlugin::initialize() {
    if (!mInitialized) {
        CHECK(cublasCreate(&mCublas));
        
        CHECK(cublasSetMathMode(mCublas, CUBLAS_TENSOR_OP_MATH));
        
        CHECK(cudaStreamCreate(&mStreamh));
            
        
        mWeights_d = (void**)malloc(mNumLayers * sizeof(void*));
        
        for (int i = 0; i < mNumLayers; i++) {        
            size_t dataTypeSize = 0;
            if (mDataType == DataType::kHALF) {
                dataTypeSize = sizeof(half);
            }
            
            size_t sz = 4 * mHiddenSize * ((i == 0 ? mInputSize : mHiddenSize) + mHiddenSize) * dataTypeSize;
            
            CHECK(cudaMalloc(&mWeights_d[i], sz));
    
            CHECK(cudaMemcpy(mWeights_d[i], mWeights_h[i], sz, cudaMemcpyHostToDevice));        
        }
        
        mBias_d = (void**)malloc(mNumLayers * sizeof(void*));
        
        for (int i = 0; i < mNumLayers; i++) {        
            size_t dataTypeSize = 0;
            if (mDataType == DataType::kHALF) {
                dataTypeSize = sizeof(half);
            }
            
            size_t sz = 8 * mHiddenSize * dataTypeSize;
            CHECK(cudaMalloc(&mBias_d[i], sz));
            
            CHECK(cudaMemcpy(mBias_d[i], mBias_h[i], sz, cudaMemcpyHostToDevice));        
           
        }        
        
        
        mWorkSpace_d = NULL;// CHECK(cudaMalloc(&mWorkSpace_d, getWorkspaceSize()));
    }

    return cudaSuccess;
}

void RNNTDecoderPlugin::terminate() {
    if (mCublas) {            
        CHECK(cublasDestroy(mCublas));
        mCublas = nullptr;
    }
    
    if (mStreamh) {
        CHECK(cudaStreamDestroy(mStreamh));
        mStreamh = nullptr;
    }
            
    if (mWeights_d) {
        for (int i = 0; i < mNumLayers; i++) {           
            if (mWeights_d[i]) {                
                cudaFree(mWeights_d[i]);
                mWeights_d[i] = nullptr;
            }
        }
        free(mWeights_d);
        mWeights_d = nullptr;
    }
    
    if (mBias_d) {
        for (int i = 0; i < mNumLayers; i++) {           
            if (mBias_d[i]) {                
                cudaFree(mBias_d[i]);
                mBias_d[i] = nullptr;
            }
        }
        free(mBias_d);
        mBias_d = nullptr;
    }
    
    if (!mWorkSpace_d) {
        cudaFree(mWorkSpace_d);
        mWorkSpace_d = nullptr;
    }
}

/* size_t RNNTDecoderPlugin::getWorkspaceSize(int maxBatchSize) const {
    size_t size = 0;
    
    // tmp_io
    size += mNumLayers * mInputSize * maxBatchSize * sizeof(half);
    
    // tmp_i
    size += mHiddenSize * maxBatchSize * 4 * sizeof(half);
    
    // tmp_h
    size += mNumLayers * mHiddenSize * maxBatchSize * 4 * sizeof(half);
    

    return size;
}  */


size_t RNNTDecoderPlugin::getWorkspaceSize(const PluginTensorDesc *inputs, int nbInputs, const PluginTensorDesc *outputs, int nbOutputs) const {
    size_t size = 0;
    
    int batchSize = inputs[0].dims.d[0];
    
    // printf("getWorkspaceSize batchSize %d\n", batchSize);
    
    // tmp_io
    size += mNumLayers * mHiddenSize * batchSize * sizeof(half);
    
    // tmp_i
    size += mHiddenSize * batchSize * 4 * sizeof(half);
    
    // tmp_h
    size += mNumLayers * mHiddenSize * batchSize * 4 * sizeof(half);
    

    return size;
}

// int RNNTDecoderPlugin::enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) {
int RNNTDecoderPlugin::enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) {
    int batchSize = inputDesc[0].dims.d[0];
    
    int effectiveBatch = batchSize;
    

    void *tmp_io = NULL;
    void *tmp_i = NULL; 
    void *tmp_h = NULL; 
    
    tmp_io = workspace;
    tmp_i = (void*)((char*)(tmp_io) + mNumLayers * mHiddenSize * effectiveBatch * sizeof(half));
    tmp_h = (void*)((char*)(tmp_i) + mHiddenSize * effectiveBatch * 4 * sizeof(half));
    
    cudaEvent_t event;
    CHECK(cudaEventCreate(&event, cudaEventDisableTiming));
    CHECK(cudaEventRecord(event, stream));  
    CHECK(cudaStreamWaitEvent(mStreamh, event, 0));
    CHECK(cudaEventDestroy(event));
   
    if (mDataType == nvinfer1::DataType::kHALF) {
        decoderStep<half, CUDA_R_16F, half, CUDA_R_16F, half>
                (mHiddenSize, 
                 mInputSize,
                 effectiveBatch, 
                 1,
                 mNumLayers,
                 this->mCublas,
                 (half*)inputs[0], // x 
                 (half*)inputs[1], // hx, 
                 (half*)inputs[2], // cx, 
                 (half**)mWeights_d,
                 (half**)mBias_d, // bias
                 (half*)outputs[0], // y, 
                 (half*)outputs[1], // hy, 
                 (half*)outputs[2], // cy,
                 (half*)tmp_io,
                 (half*)tmp_i,
                 (half*)tmp_h,
                 stream,
                 mStreamh);
    }
    
    
    return 0;
}

size_t RNNTDecoderPlugin::getSerializationSize() const {    
    size_t sz = sizeof(mNumLayers) + sizeof(mHiddenSize) + sizeof(mInputSize) + sizeof(mDataType);
    
    // Weights
    for (int i = 0; i < mNumLayers; i++) {
        size_t dataTypeSize = 0;
        if (mDataType == DataType::kHALF) {
            dataTypeSize = sizeof(half);
        }
       
        sz += 4 * mHiddenSize * ((i == 0 ? mInputSize : mHiddenSize) + mHiddenSize) * dataTypeSize;
    }
    
    // Bias
    for (int i = 0; i < mNumLayers; i++) {
        size_t dataTypeSize = 0;
        if (mDataType == DataType::kHALF) {
            dataTypeSize = sizeof(half);
        }
       
        sz += 8 * mHiddenSize * dataTypeSize;
    }
    
    return sz;
}

void RNNTDecoderPlugin::serialize(void* buffer) const {
    char *d = static_cast<char*>(buffer);
    // Use maybe_unused attribute when updating to CUDA_STANDARD C++17
    #ifndef NDEBUG
    auto d_start = d;
    #endif
    
    write<int>(d, mNumLayers);
    write<int>(d, mHiddenSize);        
    write<int>(d, mInputSize);
    write<nvinfer1::DataType>(d, mDataType);
    
    
    for (int i = 0; i < mNumLayers; i++) {        
        size_t dataTypeSize = 0;
        if (mDataType == DataType::kHALF) {
            dataTypeSize = sizeof(half);
        }
        
        size_t sz = 4 * mHiddenSize * ((i == 0 ? mInputSize : mHiddenSize) + mHiddenSize) * dataTypeSize;

        memcpy(d, mWeights_h[i], sz);
        d += sz;
    }

    for (int i = 0; i < mNumLayers; i++) {        
        size_t dataTypeSize = 0;
        if (mDataType == DataType::kHALF) {
            dataTypeSize = sizeof(half);
        }
        
        size_t sz = 8 * mHiddenSize * dataTypeSize;

        memcpy(d, mBias_h[i], sz);
        d += sz;
    }

    assert(d == d_start + getSerializationSize());
}

nvinfer1::DataType RNNTDecoderPlugin::getOutputDataType (int index, const nvinfer1::DataType *inputTypes, int nbInputs) const {
    return mDataType;
}

// bool RNNTDecoderPlugin::isOutputBroadcastAcrossBatch (int outputIndex, const bool *inputIsBroadcasted, int nbInputs) const {
    // return false;
// }

// bool RNNTDecoderPlugin::canBroadcastInputAcrossBatch (int inputIndex) const {
    // return inputIndex >= 2 * mNumLayers + 2;
// }

template <typename T>
void RNNTDecoderPlugin::write(char*& buffer, const T& val) const
{
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
}

template <typename T>
void RNNTDecoderPlugin::read(const char*& buffer, T& val) const
{
    val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
}

const char* RNNTDecoderPluginCreator::getPluginName() const {
    return "RNNTDecoderPlugin";
}

const char* RNNTDecoderPluginCreator::getPluginVersion() const {
    return "1";
}

const PluginFieldCollection* RNNTDecoderPluginCreator::getFieldNames() {
    return nullptr;        
}

void RNNTDecoderPluginCreator::setPluginNamespace(const char* libNamespace) {
    mNamespace = libNamespace;
}

const char* RNNTDecoderPluginCreator::getPluginNamespace() const {
    return mNamespace.c_str();
}

IPluginV2DynamicExt * RNNTDecoderPluginCreator::createPlugin(const char *name, const PluginFieldCollection *fc) {
    return new RNNTDecoderPlugin(fc);        
}

IPluginV2DynamicExt * RNNTDecoderPluginCreator::deserializePlugin(const char *name, const void *serialData, size_t serialLength) {
    return new RNNTDecoderPlugin(serialData, serialLength);        
}

