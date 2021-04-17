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

#include "select3Plugin.h"

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
using nvinfer1::plugin::RNNTSelectPlugin;
using nvinfer1::plugin::RNNTSelectPluginCreator;

REGISTER_TENSORRT_PLUGIN(RNNTSelectPluginCreator);

__global__ void select3(int batchSize, 
                        int size1, 
                        int size2, 
                        int size3, 
                        bool* input_select, 
                        half* input0_hidden, 
                        half* input1_hidden, 
                        half* input0_cell, 
                        half* input1_cell, 
                        int32_t* input0_winner, 
                        int32_t* input1_winner, 
                        half* isel_hidden, 
                        half* isel_cell, 
                        int32_t* isel_winner) {
                            
    int element = blockIdx.x * blockDim.x + threadIdx.x;
    int example = blockIdx.y * blockDim.y + threadIdx.y;
                            
    if (example >= batchSize) return;
                            
    bool sel = input_select[example];

    if (!sel) return;
    
    if (element < size1) {
        isel_hidden[example * size1 + element] = input0_hidden[example * size1 + element];
    }

    if (element < size2) {
        isel_cell[example * size2 + element] = input0_cell[example * size2 + element];
    }

    if (element < size3) {
        isel_winner[example * size3 + element] = input0_winner[example * size3 + element];
    }
}



RNNTSelectPlugin::RNNTSelectPlugin(const PluginFieldCollection *fc) {
}

RNNTSelectPlugin::RNNTSelectPlugin(const void* data, size_t length) {
}

const char* RNNTSelectPlugin::getPluginType() const {
    return "RNNTSelectPlugin";
}

const char* RNNTSelectPlugin::getPluginVersion() const {
    return "1";
}

void RNNTSelectPlugin::setPluginNamespace(const char* libNamespace) {
    mNamespace = libNamespace;
}

const char* RNNTSelectPlugin::getPluginNamespace() const {
    return mNamespace.c_str();
}

void RNNTSelectPlugin::destroy() {
    delete this;
}

IPluginV2DynamicExt * RNNTSelectPlugin::clone() const {    
    size_t sz = getSerializationSize();
    
    char *buff = (char*)malloc(getSerializationSize());

    // serialize is an assertion sanity check because SelectPlugin is sizeless
    serialize(buff);
    RNNTSelectPlugin* ret = new RNNTSelectPlugin(buff, sz);
    free(buff);
    
    return ret;
}

int RNNTSelectPlugin::getNbOutputs() const {
    return 3;
}

DimsExprs RNNTSelectPlugin::getOutputDimensions (int outputIndex, const DimsExprs *inputs, int nbInputs, IExprBuilder &exprBuilder) {
    
    assert(outputIndex >= 0 && outputIndex < this->getNbOutputs());
    assert(nbInputs == 7);
    
    return inputs[outputIndex * 2 + 1];
}

bool RNNTSelectPlugin::supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) { 
    if (inOut[pos].format != TensorFormat::kLINEAR) {
        return false;
    }
    
    if (nbInputs != 7 || nbOutputs != 3) {
        printf("Wrong input or output count %d %d\n", nbInputs, nbOutputs);
        return false;
    }

    // if (pos == 0 && inOut[pos].type != DataType::kBOOL) {
        // return false;
    // }
    if (pos == 0 && inOut[pos].type != DataType::kINT32) {
        return false;
    }
    
    if (pos >= 1 && pos < 5 && inOut[pos].type != DataType::kHALF) {
        return false;
    }
    
    if (pos >= 5 && pos < 7 && inOut[pos].type != DataType::kINT32) {
        return false;
    }
    
    if (pos >= 7 && pos < 9 && inOut[pos].type != DataType::kHALF) {
        return false;
    }
    
    if (pos == 9 && inOut[pos].type != DataType::kINT32) {
        return false;
    }
    
    return true;
}

void RNNTSelectPlugin::configurePlugin (const DynamicPluginTensorDesc *in, int nbInputs, const DynamicPluginTensorDesc *out, int nbOutputs) {
}

int RNNTSelectPlugin::initialize() {
    return cudaSuccess;
}

void RNNTSelectPlugin::terminate() {
}


size_t RNNTSelectPlugin::getWorkspaceSize(const PluginTensorDesc *inputs, int nbInputs, const PluginTensorDesc *outputs, int nbOutputs) const {
    size_t size = 0;

    return size;
}

// int RNNTSelectPlugin::enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) {
int RNNTSelectPlugin::enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) {
    int batchSize = inputDesc[0].dims.d[0];
    int size1 = inputDesc[1].dims.d[1] * inputDesc[1].dims.d[2];
    int size2 = inputDesc[3].dims.d[1] * inputDesc[3].dims.d[2];
    int size3 = inputDesc[5].dims.d[1] * inputDesc[5].dims.d[2];
    
    // Isn't there a max int somewhere? Probably.
    int maxSize = size1 > size2 ? size1 : size2;
    maxSize = maxSize > size3 ? maxSize : size3;
    
    dim3 blockDim = dim3(32, 8, 1);
    dim3 gridDim = dim3((maxSize + blockDim.x - 1) / blockDim.x, (batchSize + blockDim.y - 1) / blockDim.y, 1);
    
    
    select3 <<< gridDim, blockDim, 0, stream >>> (batchSize, 
                                                 size1, 
                                                 size2, 
                                                 size3, 
                                                 (bool*)inputs[0], 
                                                 (half*)inputs[1], 
                                                 (half*)inputs[2], 
                                                 (half*)inputs[3], 
                                                 (half*)inputs[4], 
                                                 (int32_t*)inputs[5], 
                                                 (int32_t*)inputs[6], 
                                                 (half*)outputs[0], 
                                                 (half*)outputs[1], 
                                                 (int32_t*)outputs[2]); 
    
    return 0;
}

size_t RNNTSelectPlugin::getSerializationSize() const {    
    size_t sz = 0;
    
    
    return sz;
}

void RNNTSelectPlugin::serialize(void* buffer) const {
    // Use maybe_unused attribute when updating to CUDA_STANDARD C++17
    #ifndef NDEBUG
    char *d = static_cast<char*>(buffer);
    auto *d_start = d;
    #endif
    
    assert(d == d_start + getSerializationSize());
}

nvinfer1::DataType RNNTSelectPlugin::getOutputDataType (int index, const nvinfer1::DataType *inputTypes, int nbInputs) const {
    if (index < 2) {
        return DataType::kHALF;
    }
    else {
        return DataType::kINT32;
    }
}

template <typename T>
void RNNTSelectPlugin::write(char*& buffer, const T& val) const
{
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
}

template <typename T>
void RNNTSelectPlugin::read(const char*& buffer, T& val) const
{
    val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
}

const char* RNNTSelectPluginCreator::getPluginName() const {
    return "RNNTSelectPlugin";
}

const char* RNNTSelectPluginCreator::getPluginVersion() const {
    return "1";
}

const PluginFieldCollection* RNNTSelectPluginCreator::getFieldNames() {
    return nullptr;        
}

void RNNTSelectPluginCreator::setPluginNamespace(const char* libNamespace) {
    mNamespace = libNamespace;
}

const char* RNNTSelectPluginCreator::getPluginNamespace() const {
    return mNamespace.c_str();
}

IPluginV2DynamicExt * RNNTSelectPluginCreator::createPlugin(const char *name, const PluginFieldCollection *fc) {
    return new RNNTSelectPlugin(fc);        
}

IPluginV2DynamicExt * RNNTSelectPluginCreator::deserializePlugin(const char *name, const void *serialData, size_t serialLength) {
    return new RNNTSelectPlugin(serialData, serialLength);        
}

