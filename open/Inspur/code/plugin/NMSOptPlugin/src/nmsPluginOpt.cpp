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

#include <algorithm>
#include <iostream>
#include <sstream>
#include <cstring>
#include <vector>

#include <cudnn.h>
#include "nmsPluginOpt.h"
#include "ssdOpt.h"
#include "ssdOptMacros.h"

using namespace nvinfer1;
using nvinfer1::plugin::NMSOptPluginCreator;
using nvinfer1::plugin::DetectionOutputOpt;
using nvinfer1::plugin::DetectionOutputParameters;

namespace
{
const char* NMS_OPT_PLUGIN_STATIC_VERSION{"1"};
const char* NMS_OPT_PLUGIN_DYNAMIC_VERSION{"2"};
const char* NMS_OPT_PLUGIN_NAME{"NMS_OPT_TRT"};
}

REGISTER_TENSORRT_PLUGIN(NMSOptPluginDynamicCreator);
REGISTER_TENSORRT_PLUGIN(NMSOptPluginStaticCreator);

template <class T>
PluginFieldCollection NMSOptPluginCreator<T>::mFC{};
template <class T>
std::vector<PluginField> NMSOptPluginCreator<T>::mPluginAttributes;

template <class T>
DetectionOutputOpt<T>::DetectionOutputOpt(DetectionOutputParameters params,
    bool confSoftmax, int numLayers)
    : param(params), mConfSoftmax(confSoftmax), mNumLayers(numLayers)
{
}

template <class T>
DetectionOutputOpt<T>::DetectionOutputOpt(const void* data, size_t length)
{
    const char* d = reinterpret_cast<const char *>(data);
    const char* a = d;
    param = read<DetectionOutputParameters>(d);
    mConfSoftmax = read<bool>(d);
    mNumLayers = read<int>(d);
    C1 = read<int>(d);
    C2 = read<int>(d);
    numPriors = read<int>(d);
    mFeatureSize.resize(mNumLayers);
    mNumAnchors.resize(mNumLayers);
    mBoxChannels.resize(mNumLayers);
    mConfChannels.resize(mNumLayers);
    for(int i = 0; i < mNumLayers; i++){
        mFeatureSize[i] = read<int>(d);
        mNumAnchors[i] = read<int>(d);
        mBoxChannels[i] = read<int>(d);
        mConfChannels[i] = read<int>(d);
    }
    mPacked32NCHW = read<bool>(d);
    ASSERT(d == a + length);
}

template <class T>
int DetectionOutputOpt<T>::getNbOutputs() const
{
    return 1;
}

template <class T>
int DetectionOutputOpt<T>::initialize()
{
    if (!mInitialized)
    {
        cudnnStatus_t status;
        status = cudnnCreate(&mCudnn);
        ASSERT(status == CUDNN_STATUS_SUCCESS);
        status = cudnnCreateTensorDescriptor(&mInScoreTensorDesc);
        ASSERT(status == CUDNN_STATUS_SUCCESS);
        status = cudnnCreateTensorDescriptor(&mOutScoreTensorDesc);
        ASSERT(status == CUDNN_STATUS_SUCCESS);
    }
    mInitialized = true;
    return 0;
}

template <class T>
void DetectionOutputOpt<T>::terminate()
{
    if (mInitialized)
    {
        cudnnStatus_t status;
        status = cudnnDestroyTensorDescriptor(mInScoreTensorDesc);
        ASSERT(status == CUDNN_STATUS_SUCCESS);
        status = cudnnDestroyTensorDescriptor(mOutScoreTensorDesc);
        ASSERT(status == CUDNN_STATUS_SUCCESS);
        status = cudnnDestroy(mCudnn);
        ASSERT(status == CUDNN_STATUS_SUCCESS);
    }
}

Dims DetectionOutputOptStatic::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    return DimsCHW(1, param.keepTopK*7 + 1, 1); //detections and keepCount
}

DimsExprs DetectionOutputOptDynamic::getOutputDimensions(int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder)
{
    const DimsExprs *mergedBoxConfInputs = &inputs[param.inputOrder[0]];
    DimsExprs outDims{4, {mergedBoxConfInputs[0].d[0], exprBuilder.constant(1), exprBuilder.constant(1), exprBuilder.constant(param.keepTopK * 7 + 1)}}; //detections and keepCount

    return outDims;
}

size_t DetectionOutputOptStatic::getWorkspaceSize(int maxBatchSize) const
{
    return detectionInferenceWorkspaceSize(param.shareLocation, maxBatchSize, C1, C2, param.numClasses, numPriors, param.topK, DataType::kFLOAT, DataType::kFLOAT);
}

size_t DetectionOutputOptDynamic::getWorkspaceSize(const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs, int nbOutputs) const
{
    int batchSize = inputs[param.inputOrder[0]].dims.d[0];
    return detectionInferenceWorkspaceSize(param.shareLocation, batchSize, C1, C2, param.numClasses, numPriors, param.topK, DataType::kFLOAT, DataType::kFLOAT);
}

template<class T>
int DetectionOutputOpt<T>::enqueueBase(int batchSize, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream)
{
    const void* const* locData = &inputs[param.inputOrder[0]];
    const void* const* confData = &inputs[param.inputOrder[1]];
    const void* priorData = inputs[param.inputOrder[2]];
    void* topDetections = outputs[0];

    ssdStatus_t status = detectionInferenceOpt(stream,
                                            batchSize,
                                            C1,
                                            C2,
                                            param.shareLocation,
                                            param.varianceEncodedInTarget,
                                            param.backgroundLabelId,
                                            numPriors,
                                            param.numClasses,
                                            param.topK,
                                            param.keepTopK,
                                            param.confidenceThreshold,
                                            param.nmsThreshold,
                                            param.codeType,
                                            DataType::kFLOAT,
                                            locData,
                                            priorData,
                                            DataType::kFLOAT,
                                            confData,
                                            topDetections,
                                            workspace,
                                            param.isNormalized,
                                            param.confSigmoid, 
                                            mConfSoftmax,
                                            mNumLayers,
                                            mFeatureSize.data(),
                                            mNumAnchors.data(),
                                            mBoxChannels.data(),
                                            mConfChannels.data(),
                                            mPacked32NCHW,
                                            mCudnn,
                                            mInScoreTensorDesc,
                                            mOutScoreTensorDesc);
    ASSERT(status == STATUS_SUCCESS);
    return 0;
}

int DetectionOutputOptStatic::enqueue(
    int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
{
    return enqueueBase(batchSize, inputs, outputs, workspace, stream);
}

int DetectionOutputOptDynamic::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs,
        void* const* outputs, void* workspace, cudaStream_t stream)
{
    return enqueueBase(inputDesc[param.inputOrder[0]].dims.d[0], inputs, outputs, workspace, stream);
}

template <class T>
size_t DetectionOutputOpt<T>::getSerializationSize() const
{
    // DetectionOutputParameters, mConfSoftmax, mNumLayers,C1,C2,numPriors
    return sizeof(DetectionOutputParameters) + sizeof(int) * 4 + sizeof(bool)
        //std::vector<int> mFeatureSize
        //std::vector<int> mNumAnchors
        //std::vector<int> mBoxChannels
        //std::vector<int> mConfChannels
        + sizeof(int)*mNumLayers*4
        //mPacked32NCHW
        + sizeof(bool);
}

template <class T>
void DetectionOutputOpt<T>::serialize(void* buffer) const
{
    char *d = reinterpret_cast<char *>(buffer), *a = d;
    write(d, param);
    write(d, mConfSoftmax);
    write(d, mNumLayers);
    write(d, C1);
    write(d, C2);
    write(d, numPriors);
    for(int i = 0; i < mNumLayers; i++) {
       write(d, mFeatureSize[i]);
       write(d, mNumAnchors[i]);
       write(d, mBoxChannels[i]);
       write(d, mConfChannels[i]);
    }
    write(d, mPacked32NCHW);
    ASSERT(d == a + getSerializationSize());
}

template <class T>
void DetectionOutputOpt<T>::configurePluginBase(const PluginTensorDesc* in, int nbInputs, const PluginTensorDesc* out, int nbOutputs)
{
    constexpr int NB_DIMS = std::is_same<T, IPluginV2IOExt>::value ? 3 : 4;
    constexpr int C_DIM = NB_DIMS - 3;
    constexpr int H_DIM = NB_DIMS - 2;
    constexpr int W_DIM = NB_DIMS - 1;

    ASSERT(out && nbOutputs == 1);
    ASSERT(out[0].dims.nbDims == NB_DIMS);

    ASSERT(in && nbInputs == mNumLayers * 2 + 1); //mNumLayers each for conf/box + 1 for prior

    //prior
    ASSERT(in[param.inputOrder[2]].dims.nbDims == 3);
    numPriors = in[param.inputOrder[2]].dims.d[1] / 4;

    // box and conf are in a single input tensor, merged in C dims
    // box channels come first, then conf channels
    // they should always have same H/W dims
    const PluginTensorDesc* mergedBoxConfInputs = &in[param.inputOrder[0]];

    C1 = 0;
    C2 = 0;
    mFeatureSize.resize(mNumLayers);
    mNumAnchors.resize(mNumLayers);
    mBoxChannels.resize(mNumLayers);
    mConfChannels.resize(mNumLayers);

    // Math to infer #channels only cover shareLocation = true:
    ASSERT(param.shareLocation);
    for(int i = 0; i < mNumLayers; i++) {
        const Dims &mergedBoxConfInputDims = mergedBoxConfInputs[i].dims;
        ASSERT(mergedBoxConfInputDims.nbDims == NB_DIMS);

        // Math to infer #channels:
        // box channels = numAnchors * 4
        // conf channels = numAnchors * numClasses
        // merged channel = box channels + conf channels = numAnchors * (numClasses + 4)
        int mergedChannels = mergedBoxConfInputDims.d[C_DIM];
        ASSERT(mergedChannels % (param.numClasses + 4) == 0);
        mNumAnchors[i] = mergedChannels / (param.numClasses + 4);
        mBoxChannels[i] = mNumAnchors[i] * 4;
        mConfChannels[i] = mNumAnchors[i] * param.numClasses;
        int flattenBoxInput = mBoxChannels[i] * mergedBoxConfInputDims.d[H_DIM] * mergedBoxConfInputDims.d[W_DIM];
        int flattenConfInput = mConfChannels[i] * mergedBoxConfInputDims.d[H_DIM] * mergedBoxConfInputDims.d[W_DIM];
        C1 += flattenBoxInput;
        C2 += flattenConfInput;

        ASSERT(mBoxChannels[i] / 4 == mConfChannels[i] / param.numClasses);

        mFeatureSize[i] = mergedBoxConfInputDims.d[H_DIM]; //ASSERT H=W
    }

    const int numLocClasses = param.shareLocation ? 1 : param.numClasses;
    ASSERT(numPriors * numLocClasses * 4 == C1);
    ASSERT(numPriors * param.numClasses == C2);

    //Check types and format
    for(int i = 0; i < mNumLayers; i++) {
        const auto inputType = mergedBoxConfInputs[i].type;
        const auto inputFormat = mergedBoxConfInputs[i].format;
        if(i == 0) {
            ASSERT(inputType == DataType::kFLOAT && (inputFormat == TensorFormat::kCHW32 || inputFormat == TensorFormat::kLINEAR));
            mPacked32NCHW = (inputFormat == TensorFormat::kCHW32);
        }
        else {
            ASSERT(inputType == DataType::kFLOAT && inputFormat == mergedBoxConfInputs[0].format);
        }

    }
    ASSERT(in[param.inputOrder[2]].type == DataType::kFLOAT && in[param.inputOrder[2]].format == TensorFormat::kLINEAR);
    ASSERT(out[0].type == DataType::kFLOAT && out[0].format == TensorFormat::kLINEAR);
}

void DetectionOutputOptStatic::configurePlugin(const PluginTensorDesc* in, int nbInputs, const PluginTensorDesc* out, int nbOutputs)
{
    configurePluginBase(in, nbInputs, out, nbOutputs);
}

void DetectionOutputOptDynamic::configurePlugin(const DynamicPluginTensorDesc* in, int nbInputs, const DynamicPluginTensorDesc* out, int nbOutputs)
{
    std::vector<PluginTensorDesc> inDesc(nbInputs);
    std::transform(in, in + nbInputs, inDesc.begin(), [](const DynamicPluginTensorDesc& i){ return i.desc; });
    std::vector<PluginTensorDesc> outDesc(nbOutputs);
    std::transform(out, out + nbOutputs, outDesc.begin(), [](const DynamicPluginTensorDesc& i){ return i.desc; });
    configurePluginBase(inDesc.data(), nbInputs, outDesc.data(), nbOutputs);
}

template <class T>
bool DetectionOutputOpt<T>::supportsFormatCombinationBase(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const
{
    ASSERT(nbInputs == mNumLayers * 2 + 1); //mNumLayers each for conf/box + 1 for prior
    ASSERT(nbOutputs == 1);

    bool rtn;
    if((pos >= param.inputOrder[0] && pos < param.inputOrder[0] + mNumLayers) //boxInput
        || (pos >= param.inputOrder[1] && pos < param.inputOrder[1] + mNumLayers)) //confInput
    {
        //use fp32 NC/32HW32
        rtn = inOut[pos].type == DataType::kFLOAT && (inOut[pos].format == TensorFormat::kCHW32 || inOut[pos].format == TensorFormat::kLINEAR);
        if(param.inputOrder[0] < param.inputOrder[1]) {
            rtn &= (inOut[pos].format == inOut[param.inputOrder[0]].format);
        }
        else {
            rtn &= (inOut[pos].format == inOut[param.inputOrder[1]].format);
        }
    }
    else if(pos == param.inputOrder[2]) //prior, just uses fp32 NCHW
    {
        rtn = inOut[pos].type == DataType::kFLOAT && inOut[pos].format == TensorFormat::kLINEAR;
    }
    else {
        ASSERT(pos == nbInputs); // output
        rtn = inOut[pos].type == DataType::kFLOAT && inOut[pos].format == TensorFormat::kLINEAR;
    }

    return rtn;
}

bool DetectionOutputOptStatic::supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const
{
    return supportsFormatCombinationBase(pos, inOut, nbInputs, nbOutputs);
}

bool DetectionOutputOptDynamic::supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs)
{
    return supportsFormatCombinationBase(pos, inOut, nbInputs, nbOutputs);
}

template <class T>
DataType DetectionOutputOpt<T>::getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const 
{
    ASSERT(index == 0);
    ASSERT(nbInputs == mNumLayers * 2 + 1);
    return DataType::kFLOAT;
}

template <class T>
const char* DetectionOutputOpt<T>::getPluginType() const { return NMS_OPT_PLUGIN_NAME; }

const char* DetectionOutputOptStatic::getPluginVersion() const { return NMS_OPT_PLUGIN_STATIC_VERSION; }
const char* DetectionOutputOptDynamic::getPluginVersion() const { return NMS_OPT_PLUGIN_DYNAMIC_VERSION; }

template <class T>
void DetectionOutputOpt<T>::destroy() { delete this; }

template <class T>
template <class Derived>
T* DetectionOutputOpt<T>::cloneBase() const
{
    Derived* plugin = new Derived(*static_cast<const Derived*>(this));
    return plugin;
}

IPluginV2Ext* DetectionOutputOptStatic::clone() const
{
    return cloneBase<DetectionOutputOptStatic>();
}

IPluginV2DynamicExt* DetectionOutputOptDynamic::clone() const
{
    return cloneBase<DetectionOutputOptDynamic>();
}

bool DetectionOutputOptStatic::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
{
    return false;
}

bool DetectionOutputOptStatic::canBroadcastInputAcrossBatch(int inputIndex) const 
{
    return inputIndex == param.inputOrder[2]; // prior
}

template <class T>
NMSOptPluginCreator<T>::NMSOptPluginCreator()
{
    mPluginAttributes.emplace_back(PluginField("shareLocation", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("varianceEncodedInTarget", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("backgroundLabelId", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("numClasses", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("topK", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("keepTopK", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("confidenceThreshold", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("nmsThreshold", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("inputOrder", nullptr, PluginFieldType::kINT32, 3));
    mPluginAttributes.emplace_back(PluginField("confSigmoid", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("confSoftmax", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("isNormalized", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("codeType", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("numLayers", nullptr, PluginFieldType::kINT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

template <class T>
const char* NMSOptPluginCreator<T>::getPluginName() const
{
    return NMS_OPT_PLUGIN_NAME;
}

template <class T>
const char* NMSOptPluginCreator<T>::getPluginVersion() const
{
    if (std::is_same<T, DetectionOutputOptStatic>::value)
    {
        return NMS_OPT_PLUGIN_STATIC_VERSION;
    }
    if (std::is_same<T, DetectionOutputOptDynamic>::value)
    {
        return NMS_OPT_PLUGIN_DYNAMIC_VERSION;
    }
    return "";
}

template <class T>
const PluginFieldCollection* NMSOptPluginCreator<T>::getFieldNames()
{
    return &mFC;
}

template <class T>
IPluginV2* NMSOptPluginCreator<T>::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    const PluginField* fields = fc->fields;
    //Default init values for TF SSD network
    params.codeType = CodeTypeSSD::TF_CENTER;
    params.inputOrder[0] = 0;
    params.inputOrder[1] = 7;
    params.inputOrder[2] = 6;

    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "shareLocation"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            params.shareLocation = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "varianceEncodedInTarget"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            params.varianceEncodedInTarget = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "backgroundLabelId"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            params.backgroundLabelId = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "numClasses"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            params.numClasses = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "topK"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            params.topK = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "keepTopK"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            params.keepTopK = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "confidenceThreshold"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            params.confidenceThreshold = static_cast<float>(*(static_cast<const float*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "nmsThreshold"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            params.nmsThreshold = static_cast<float>(*(static_cast<const float*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "confSigmoid"))
        {
            params.confSigmoid = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "confSoftmax"))
        {
            mConfSoftmax = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "isNormalized"))
        {
            params.isNormalized = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "inputOrder"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            int size = fields[i].length;
            const int* o = static_cast<const int*>(fields[i].data);
            for (int j = 0; j < size; j++)
            {
                params.inputOrder[j] = *o;
                o++;
            }
        }
        else if (!strcmp(attrName, "codeType"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            params.codeType = static_cast<CodeTypeSSD>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "numLayers"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            mNumLayers = *(static_cast<const bool*>(fields[i].data));
        }

    }

    return new T(params, mConfSoftmax, mNumLayers);
}

template <class T>
IPluginV2* NMSOptPluginCreator<T>::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    //This object will be deleted when the network is destroyed, which will
    //call NMS::destroy()
    return new T(serialData, serialLength);
}

// Can be removed - need it in order to use its contructor directly instead of the plugin creator
template class nvinfer1::plugin::DetectionOutputOpt<nvinfer1::IPluginV2DynamicExt>;
template class nvinfer1::plugin::DetectionOutputOpt<nvinfer1::IPluginV2IOExt>;
