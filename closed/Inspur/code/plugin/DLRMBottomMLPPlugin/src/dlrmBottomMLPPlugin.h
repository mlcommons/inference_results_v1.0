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

#include "NvInferPlugin.h"
#include <cassert>
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <map>
#include <cuda.h>

using namespace nvinfer1::plugin;
namespace nvinfer1
{
namespace plugin
{

class DLRMBottomMLPPlugin : public IPluginV2DynamicExt
{
public:
    // Constructor, Destructor
    DLRMBottomMLPPlugin(
        int inputChannels,
        const std::vector<std::vector<float>>& weights,
        const std::vector<std::vector<float>>& biases,
        const std::vector<float>& dynamicRanges);
    DLRMBottomMLPPlugin(const void* data, size_t length);
    ~DLRMBottomMLPPlugin() override = default;

    // IPluginV2Ext fields
    int getNbOutputs() const override;
    DataType getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const override;
    size_t getSerializationSize() const override;
    void serialize(void* buffer) const override;
    void destroy() override;
    void setPluginNamespace(const char* libNamespace) override { mNamespace = libNamespace; }
    const char* getPluginNamespace() const override { return mNamespace.c_str(); }
    IPluginV2DynamicExt* clone() const override;
    int initialize() override;
    void terminate() override;
    const char* getPluginType() const override;
    const char* getPluginVersion() const override;

    // IPluginV2DynamicExt new fields
    DimsExprs getOutputDimensions(int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) override;
    bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) override;
    size_t getWorkspaceSize(const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs, int nbOutputs) const override;
    void configurePlugin(const DynamicPluginTensorDesc* in, int nbInputs, const DynamicPluginTensorDesc* out, int nbOutputs) override;
    int enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs,
        void* const* outputs, void* workspace, cudaStream_t stream) override;

private:
    void runFusedBottomMLP(
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
        int batchSize);

    static std::vector<int8_t> shuffleWeights(
        const std::vector<int8_t>& originalWeights,
        int rows,
        int layerId);

private:
    std::string mNamespace;

    bool mInitialized;
    std::vector<int8_t *> mDeviceWeights;
    std::vector<float *> mDeviceBiases;
    CUmodule mModule;
    CUfunction mKernel;

    int mInputChannels;
    std::vector<std::vector<int8_t>> mWeights;
    std::vector<float> mWeightScales;
    std::vector<std::vector<float>> mBiases;
    std::vector<float> mActivationScales;

    static const char * mBottomMLPFusionKernelCode;
    static const int mWarpsInThreadblock;
};

class DLRMBottomMLPPluginCreator : public IPluginCreator
{
public:
    DLRMBottomMLPPluginCreator();

    ~DLRMBottomMLPPluginCreator() override = default;

    const char* getPluginName() const override;

    const char* getPluginVersion() const override;

    const PluginFieldCollection* getFieldNames() override;

    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) override;

    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;

    void setPluginNamespace(const char* libNamespace) override { mNamespace = libNamespace; }

    const char* getPluginNamespace() const override { return mNamespace.c_str(); }

private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
    std::string mNamespace;
};

} // namespace plugin
} // namespace nvinfer1
