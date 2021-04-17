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

#ifndef GNMT_DECODER_PLUGIN_H
#define GNMT_DECODER_PLUGIN_H

#include <NvInfer.h>
#include "NvInferPlugin.h"
#include <cublas_v2.h>
#include <cublasLt.h>


#include "cuda_runtime_api.h"
#include "cuda_runtime.h"

#include <stdio.h>
#include <assert.h>

#include <iostream>
#include <string>

namespace nvinfer1
{

namespace plugin
{

class RNNTSelectPlugin : public IPluginV2DynamicExt  {
public:
    RNNTSelectPlugin(const PluginFieldCollection *fc);

    // create the plugin at runtime from a byte stream
    RNNTSelectPlugin(const void* data, size_t length);
    ~RNNTSelectPlugin() override = default;

    const char *getPluginType() const override;
    
    const char *getPluginVersion() const override;
    
    void setPluginNamespace(const char* libNamespace) override;

    const char* getPluginNamespace() const override;
    
    void destroy() override;
    
    IPluginV2DynamicExt * clone() const override;

    int getNbOutputs() const override;

    // Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;
    DimsExprs getOutputDimensions (int outputIndex, const DimsExprs *inputs, int nbInputs, IExprBuilder &exprBuilder) override;

    bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) override;

    // void configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput) override;
    void configurePlugin(const DynamicPluginTensorDesc *in, int nbInputs, const DynamicPluginTensorDesc *out, int nbOutputs) override;
    
    // void configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims,
                                 // int nbOutputs, const DataType* inputTypes, const DataType* outputTypes,
                                 // const bool* inputIsBroadcast, const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) override;
    
    int initialize() override;

    virtual void terminate() override;

    // virtual size_t getWorkspaceSize(int maxBatchSize) const override;
    virtual size_t getWorkspaceSize(const PluginTensorDesc *inputs, int nbInputs, const PluginTensorDesc *outputs, int nbOutputs) const override;
    
    // virtual int enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override;
    virtual int enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) override;

    size_t getSerializationSize() const override;

    void serialize(void* buffer) const override;
    
    
    nvinfer1::DataType getOutputDataType (int index, const nvinfer1::DataType *inputTypes, int nbInputs) const override;
    
    // bool isOutputBroadcastAcrossBatch (int outputIndex, const bool *inputIsBroadcasted, int nbInputs) const override;
    
    // bool canBroadcastInputAcrossBatch (int inputIndex) const override;

    template <typename T>
    void write(char*& buffer, const T& val) const;

    template <typename T>
    void read(const char*& buffer, T& val) const;
    
    
protected:
    // To prevent compiler warnings.
    using nvinfer1::IPluginV2DynamicExt::getOutputDimensions;
    using nvinfer1::IPluginV2DynamicExt::isOutputBroadcastAcrossBatch;
    using nvinfer1::IPluginV2DynamicExt::canBroadcastInputAcrossBatch;
    using nvinfer1::IPluginV2DynamicExt::supportsFormat;
    using nvinfer1::IPluginV2DynamicExt::configurePlugin;
    using nvinfer1::IPluginV2DynamicExt::getWorkspaceSize;
    using nvinfer1::IPluginV2DynamicExt::enqueue;

private:
    int mDevice;
    int mSMVersionMajor;
    int mSMVersionMinor;
    
    std::string mNamespace;
};

class RNNTSelectPluginCreator : public IPluginCreator {
public:
    RNNTSelectPluginCreator() = default;

    ~RNNTSelectPluginCreator() override = default;

    const char *getPluginName() const override;
    
    const char *getPluginVersion() const override;
    
    const PluginFieldCollection *getFieldNames() override;
    
    void setPluginNamespace(const char* libNamespace) override;

    const char* getPluginNamespace() const override;
    
    IPluginV2DynamicExt  *createPlugin(const char *name, const PluginFieldCollection *fc) override;
    
    IPluginV2DynamicExt  *deserializePlugin(const char *name, const void *serialData, size_t serialLength) override;

private:
    std::string mNamespace;
};

} // namespace plugin
} // namespace nvinfer1

#endif // GNMT_DECODER_PLUGIN_H
