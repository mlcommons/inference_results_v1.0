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

using namespace nvinfer1::plugin;
namespace nvinfer1
{
namespace plugin
{

// There are two versions of DetectionOutputOpt.
// Version 1 is based on IPluginV2IOExt, to be used with Implicit batch.
// Version 2 is based on IPluginV2DynamicExt, to be used with Explicit batch.
template<class T>
class DetectionOutputOpt : public T
{
public:
    // Constructor, Destructor
    DetectionOutputOpt(DetectionOutputParameters param, bool confSoftmax, int numLayers);
    DetectionOutputOpt(const void* data, size_t length);
    ~DetectionOutputOpt() override = default;

    // IPluginV2Ext fields
    int getNbOutputs() const override;
    DataType getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const override;
    size_t getSerializationSize() const override;
    void serialize(void* buffer) const override;
    void destroy() override;
    void setPluginNamespace(const char* libNamespace) override { mNamespace = libNamespace; }
    const char* getPluginNamespace() const override { return mNamespace.c_str(); }
    int initialize() override;
    void terminate() override;
    const char* getPluginType() const override;

    // Shared methods by derived classes
    bool supportsFormatCombinationBase(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const;
    template <class Derived>
    T* cloneBase() const;
    int enqueueBase(int batchSize, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream);
    void configurePluginBase(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput);

protected:
    DetectionOutputParameters param;
    bool mConfSoftmax; 
    int mNumLayers;
    int C1, C2, numPriors;
    std::vector<int> mFeatureSize;
    std::vector<int> mNumAnchors;
    std::vector<int> mBoxChannels;
    std::vector<int> mConfChannels;
    bool mPacked32NCHW;
    std::string mNamespace;
    bool mInitialized{false};
    cudnnHandle_t mCudnn;
    cudnnTensorDescriptor_t mInScoreTensorDesc;
    cudnnTensorDescriptor_t mOutScoreTensorDesc;
};

class DetectionOutputOptStatic : public DetectionOutputOpt<IPluginV2IOExt>
{
public:
    DetectionOutputOptStatic(DetectionOutputParameters param, bool confSoftmax, int numLayers)
        : DetectionOutputOpt(param, confSoftmax, numLayers){};
    DetectionOutputOptStatic(const void* data, size_t length)
        : DetectionOutputOpt(data, length){};
    ~DetectionOutputOptStatic() override = default;

    const char* getPluginVersion() const override;
    IPluginV2Ext* clone() const override;

    // IPluginV2Ext methods, deprecated in IPluginV2DynamicExt
    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;
    size_t getWorkspaceSize(int maxBatchSize) const override;
    bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const override;
    bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const override;
    bool canBroadcastInputAcrossBatch(int inputIndex) const override;
    int enqueue(
        int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override;

    // IPluginV2IOExt methods
    void configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput) override;
};

class DetectionOutputOptDynamic : public DetectionOutputOpt<IPluginV2DynamicExt>
{
public:
    DetectionOutputOptDynamic(DetectionOutputParameters param, bool confSoftmax, int numLayers)
        : DetectionOutputOpt(param, confSoftmax, numLayers){};
    DetectionOutputOptDynamic(const void* data, size_t length)
        : DetectionOutputOpt(data, length){};
    ~DetectionOutputOptDynamic() override = default;

    const char* getPluginVersion() const override;
    IPluginV2DynamicExt* clone() const override;

    // IPluginV2DynamicExt methods
    DimsExprs getOutputDimensions(
        int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) override;
    bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) override;
    void configurePlugin(
        const DynamicPluginTensorDesc* in, int nbInputs, const DynamicPluginTensorDesc* out, int nbOutputs) override;
    size_t getWorkspaceSize(
        const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs, int nbOutputs) const override;
    int enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs,
        void* const* outputs, void* workspace, cudaStream_t stream) override;
};

template <class T>
class NMSOptPluginCreator : public IPluginCreator
{
public:
    NMSOptPluginCreator();

    ~NMSOptPluginCreator() override = default;

    const char* getPluginName() const override;

    const char* getPluginVersion() const override;

    const PluginFieldCollection* getFieldNames() override;

    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) override;

    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;

    void setPluginNamespace(const char* libNamespace) override { mNamespace = libNamespace; }

    const char* getPluginNamespace() const override { return mNamespace.c_str(); }

private:
    static PluginFieldCollection mFC;
    DetectionOutputParameters params;
    bool mConfSoftmax;
    int mNumLayers;
    static std::vector<PluginField> mPluginAttributes;
    std::string mNamespace;
};

using NMSOptPluginStaticCreator = NMSOptPluginCreator<DetectionOutputOptStatic>;
using NMSOptPluginDynamicCreator = NMSOptPluginCreator<DetectionOutputOptDynamic>;

} // namespace plugin
} // namespace nvinfer1
