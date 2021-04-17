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
#include <stdexcept>

#include "conv3d1x1x1K4Plugin.h"

using namespace nvinfer1;
using nvinfer1::plugin::conv3D1X1X1K4Plugin;
using nvinfer1::plugin::conv3D1X1X1K4PluginCreator;

////////////////////////////////////////////////////////////////////////////////////////////////////

#define CHECK_CUDA(call) do                                                                    \
    {                                                                                          \
    cudaError_t status_ = call;                                                                \
    if( status_ != cudaSuccess )                                                               \
    {                                                                                          \
        fprintf(stderr, "CUDA Error at line %d: %s\n", __LINE__, cudaGetErrorString(status_)); \
        exit(1);                                                                               \
        }                                                                                      \
    } while(0)
  
////////////////////////////////////////////////////////////////////////////////////////////////////
  
  #define CHECK_CUDNN(call) do                                                                   \
    {                                                                                            \
    cudnnStatus_t status_ = call;                                                                \
    if( status_ != CUDNN_STATUS_SUCCESS )                                                        \
    {                                                                                            \
        fprintf(stderr, "CUDNN Error at line %d: %s\n", __LINE__, cudnnGetErrorString(status_)); \
        exit(1);                                                                                 \
        }                                                                                        \
    } while(0)
  
////////////////////////////////////////////////////////////////////////////////////////////////////

enum PluginMode 
{
    FP32_LINEAR_FP32_LINEAR_MODE, // FP32 Linear -> FP32 Linear
    INT8_CDHW32_FP16_LINEAR_MODE // INT8 NCDWH32 -> FP16 Linear
};

// Checks both intput/output format/datatypes
static PluginMode getPluginMode(const nvinfer1::PluginTensorDesc& inputDesc, const nvinfer1::PluginTensorDesc& outputDesc)
{
    if (inputDesc.format == nvinfer1::PluginFormat::kLINEAR && inputDesc.type == nvinfer1::DataType::kFLOAT && 
        outputDesc.format == nvinfer1::PluginFormat::kLINEAR && outputDesc.type == nvinfer1::DataType::kFLOAT)
    {
        return  FP32_LINEAR_FP32_LINEAR_MODE;
    }
    else if ( inputDesc.format == nvinfer1::PluginFormat::kCDHW32 && inputDesc.type == nvinfer1::DataType::kINT8 &&
        outputDesc.format == nvinfer1::PluginFormat::kLINEAR && outputDesc.type == nvinfer1::DataType::kHALF)
    {
        return  INT8_CDHW32_FP16_LINEAR_MODE;
    }
    else
    {
        ASSERT(false && "Unexpected input format");
    }
}

namespace 
{
    const char* CONV3D1X1X1K4_PLUGIN_VERSION{"1"};
    const char* CONV3D1X1X1K4_PLUGIN_NAME{"CONV3D1X1X1K4_TRT"};
}

REGISTER_TENSORRT_PLUGIN(conv3D1X1X1K4PluginCreator);

PluginFieldCollection conv3D1X1X1K4PluginCreator::mFC{};
std::vector<PluginField> conv3D1X1X1K4PluginCreator::mPluginAttributes;

conv3D1X1X1K4Plugin::conv3D1X1X1K4Plugin(
    int inputChannels,
    const std::vector<float>& weights)
    : mInitialized(false)
    , mInputChannels(inputChannels)
    , mInActivationScale(-1.0F)
    , mOutActivationScale(-1.0F)
    , mWeights(weights)
{
}

conv3D1X1X1K4Plugin::conv3D1X1X1K4Plugin(void const* serialData, size_t serialLength)
{
    deserialize_value(&serialData, &serialLength, &mInputChannels);
    deserialize_value(&serialData, &serialLength, &mWeights);
    deserialize_value(&serialData, &serialLength, &mWeightScale);
    deserialize_value(&serialData, &serialLength, &mInActivationScale);
    deserialize_value(&serialData, &serialLength, &mOutActivationScale);
}

conv3D1X1X1K4Plugin::~conv3D1X1X1K4Plugin()
{
    terminate();
}

// conv3D1X1X1K4Plugin returns one output.
int conv3D1X1X1K4Plugin::getNbOutputs() const
{
    return 1;
}

DimsExprs conv3D1X1X1K4Plugin::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder)
{
    nvinfer1::DimsExprs output(inputs[0]);
    output.d[1] =  exprBuilder.operation(DimensionOperation::kFLOOR_DIV,  *inputs[0].d[1], *exprBuilder.constant(8));

    return output;
}

int conv3D1X1X1K4Plugin::initialize()
{
    if (!mInitialized)
    {
        CHECK_CUDA(cudaMalloc(&mDeviceWeights, mWeights.size() * sizeof(float)));
        // kLINEAR
        CHECK_CUDNN(cudnnCreate(&mCudnnHandle));
        CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&mConvDesc));
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&mImageDesc));
        CHECK_CUDNN(cudnnCreateFilterDescriptor(&mFltDesc));
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&mOutDesc));

        const int padA[3] = {0, 0, 0};
        const int dilationA[3] = {1, 1, 1};
        const int convstrideA[3] = {1, 1, 1};
        CHECK_CUDNN(cudnnSetConvolutionNdDescriptor(mConvDesc, 
            3,
            padA,
            convstrideA,
            dilationA,
            CUDNN_CROSS_CORRELATION,
            CUDNN_DATA_FLOAT));

        // kCDHW32
        int device;
        CHECK_CUDA(cudaGetDevice(&device));
        cudaDeviceProp props;
        CHECK_CUDA(cudaGetDeviceProperties(&props, device));

        memset(&mContext, 0, sizeof(mContext));
        mContext.sm_count = props.multiProcessorCount;
        mContext.sm_shared_size = props.sharedMemPerMultiprocessor;
        mContext.sm_version = props.major * 100 + props.minor * 10;

        memset(&mParams, 0, sizeof(mParams));

        mInitialized = true;
    }
    return 0;
}

void conv3D1X1X1K4Plugin::terminate()
{
    if (mInitialized)
    {
        cudaFree(mDeviceWeights);

        // Release cuDNN descriptors and handle.
        cudnnDestroyConvolutionDescriptor(mConvDesc);
        cudnnDestroyTensorDescriptor(mImageDesc);
        cudnnDestroyFilterDescriptor(mFltDesc);
        cudnnDestroyTensorDescriptor(mOutDesc);
        cudnnDestroy(mCudnnHandle);

        mInitialized = false;
    }
    return;
}

size_t conv3D1X1X1K4Plugin::setCudnnDescriptors(const nvinfer1::PluginTensorDesc* inputs) const
// retursn workspace size needed by cuDNN
{
    nvinfer1::Dims input_dims = inputs[0].dims;
    const int n = input_dims.d[0];
    const int c = input_dims.d[1];
    const int d = input_dims.d[2];
    const int h = input_dims.d[3];
    const int w = input_dims.d[4];

    const int k = 4;

    const int nbDims = 5;
    const int dimA[nbDims] = {n, c, d, h, w};
    const int filterDimA[nbDims] = {k, c, 1, 1, 1};

    int strideA[nbDims];
    strideA[nbDims - 1] = 1;
    for(int dd = nbDims-2 ; dd >= 0 ; dd--) 
    {
        strideA[dd] = strideA[dd+1] * dimA[dd+1] ;
    }

    CHECK_CUDNN(cudnnSetConvolutionMathType(mConvDesc, CUDNN_DEFAULT_MATH));
    CHECK_CUDNN(cudnnSetTensorNdDescriptor(mImageDesc,
        CUDNN_DATA_FLOAT,
        nbDims,
        dimA,
        strideA)); 
    CHECK_CUDNN(cudnnSetFilterNdDescriptor(mFltDesc, 
        CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW,
        nbDims,
        filterDimA)); 
    const int outDimA[nbDims] = {n, k, d, h, w};
    int outStrideA[nbDims];
    outStrideA[nbDims - 1] = 1;
    for(int dd = nbDims-2 ; dd >= 0 ; dd--) 
    {
        outStrideA[dd] = outStrideA[dd+1] * outDimA[dd+1] ;
    }
    CHECK_CUDNN(cudnnSetTensorNdDescriptor(mOutDesc, 
        CUDNN_DATA_FLOAT,
        nbDims,
        outDimA,
        outStrideA));

    // Determine workspace sizes for the different convolutions.
    size_t workspace_sz = 0;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(mCudnnHandle,
                                                        mImageDesc,
                                                        mFltDesc,
                                                        mConvDesc,
                                                        mOutDesc,
                                                        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
                                                        &workspace_sz));
    return workspace_sz;
} 

size_t conv3D1X1X1K4Plugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs, const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const 
{
    const PluginMode pluginMode = getPluginMode(inputs[0], outputs[0]);
    if (pluginMode == FP32_LINEAR_FP32_LINEAR_MODE)
    {
        return setCudnnDescriptors(inputs);
    }
    // "INT8 NCDHW32 -> FP16 Linear" uses zero workspace.
    return 0;
}


int conv3D1X1X1K4Plugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream)
{
    ASSERT(mInitialized);
    ASSERT(mInputChannels == 32);

    const PluginMode pluginMode = getPluginMode(inputDesc[0], outputDesc[0]);

    if (pluginMode == FP32_LINEAR_FP32_LINEAR_MODE)
    {
        // use cuDNN
        CHECK_CUDNN(cudnnSetStream(mCudnnHandle, stream));

        size_t workspace_sz = setCudnnDescriptors(inputDesc);

        const float one = 1.0F, zero = 0.0F;
        CHECK_CUDNN(cudnnConvolutionForward(mCudnnHandle,
            &one,
            mImageDesc,
            inputs[0],
            mFltDesc,
            mDeviceWeights,
            mConvDesc,
            CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
            workspace,
            workspace_sz,
            &zero,
            mOutDesc,
            outputs[0]));
    }
    else if (pluginMode == INT8_CDHW32_FP16_LINEAR_MODE)
    {
        const nvinfer1::Dims input_dims = inputDesc[0].dims;
        const int n = input_dims.d[0];
        const int c = input_dims.d[1];
        const int d = input_dims.d[2];
        const int h = input_dims.d[3];
        const int w = input_dims.d[4];
    
        const int k = 4;

        mParams.gmem_a = const_cast<void *>(inputs[0]);;
        mParams.gmem_b = mDeviceWeights;
        mParams.gmem_c = outputs[0];
        mParams.img_n = n;
        mParams.img_c = c;
        mParams.img_d = d;
        mParams.img_h = h;
        mParams.img_w = w;
        mParams.flt_k = k;
    
        mParams.img_dhw = mParams.img_d * mParams.img_h * mParams.img_w;
    
        mParams.m = mParams.img_n * mParams.img_d * mParams.img_h * mParams.img_w;
        mParams.n = mParams.flt_k;
        mParams.k = mParams.img_c;
    
        mParams.lda = mParams.img_c;
        mParams.ldb = mParams.img_c;
        mParams.ldc = mParams.flt_k;

        mParams.scale = mInActivationScale * mWeightScale / mOutActivationScale;

        conv3d_1x1x1_k4_dispatch(mContext, mParams, stream);
    }

    return 0;
}

size_t conv3D1X1X1K4Plugin::getSerializationSize() const
{
    return (serialized_size(mInputChannels) +
            serialized_size(mWeights) +
            serialized_size(mWeightScale) +
            serialized_size(mInActivationScale) +
            serialized_size(mOutActivationScale));
}

void conv3D1X1X1K4Plugin::serialize(void *buffer) const
{
    serialize_value(&buffer, mInputChannels);
    serialize_value(&buffer, mWeights);
    serialize_value(&buffer, mWeightScale);
    serialize_value(&buffer, mInActivationScale);
    serialize_value(&buffer, mOutActivationScale);
}

bool conv3D1X1X1K4Plugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs)
{
    ASSERT(inOut && pos < (nbInputs + nbOutputs));

    bool support_fp32_linear = (inOut[pos].type == nvinfer1::DataType::kFLOAT
        && inOut[pos].format == nvinfer1::PluginFormat::kLINEAR
        && inOut[pos].type == inOut[0].type
        && inOut[pos].format == inOut[0].format);

    bool support_int8_cdhw32_to_fp16_linear = (pos < nbInputs) ?
    (inOut[pos].type == nvinfer1::DataType::kINT8 && inOut[pos].format == nvinfer1::PluginFormat::kCDHW32) :
    (inOut[pos].type == nvinfer1::DataType::kHALF && inOut[pos].format == nvinfer1::PluginFormat::kLINEAR)
      && inOut[0].type == nvinfer1::DataType::kINT8
      && inOut[0].format == nvinfer1::PluginFormat::kCDHW32;

    return support_fp32_linear|| support_int8_cdhw32_to_fp16_linear;
}

const char* conv3D1X1X1K4Plugin::getPluginType() const
{
    return CONV3D1X1X1K4_PLUGIN_NAME;
}

const char* conv3D1X1X1K4Plugin::getPluginVersion() const
{
    return CONV3D1X1X1K4_PLUGIN_VERSION;
}

void conv3D1X1X1K4Plugin::destroy()
{ 
    delete this;
}

IPluginV2DynamicExt* conv3D1X1X1K4Plugin::clone() const
{ 
    auto plugin = new conv3D1X1X1K4Plugin(mInputChannels, mWeights);
    plugin->setPluginNamespace(mPluginNamespace);
    plugin->initialize();
    return plugin;
}

// Set plugin namespace
void conv3D1X1X1K4Plugin::setPluginNamespace(const char* pluginNamespace)
{
    mPluginNamespace = pluginNamespace;
}

const char* conv3D1X1X1K4Plugin::getPluginNamespace() const
{
    return mPluginNamespace;
}

nvinfer1::DataType conv3D1X1X1K4Plugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    ASSERT(inputTypes && nbInputs > 0 && index == 0);

    return nvinfer1::DataType::kFLOAT;
}

void conv3D1X1X1K4Plugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs)
{
    const PluginMode pluginMode = getPluginMode(in[0].desc, out[0].desc);

    if (pluginMode == FP32_LINEAR_FP32_LINEAR_MODE)
    {
        CHECK_CUDA(cudaMemcpy(mDeviceWeights, mWeights.data(), mWeights.size() * sizeof(float), cudaMemcpyHostToDevice));
    }
    else if (pluginMode == INT8_CDHW32_FP16_LINEAR_MODE)
    {
        mInActivationScale = in[0].desc.scale;
        std::vector<int8_t> weights;
        float maxAbsVal = *std::max_element(mWeights.begin(), mWeights.end());
        float mult = 127.0F / maxAbsVal;
        mWeightScale = 1.0F / mult;
        weights.resize(mWeights.size());
        std::transform(mWeights.begin(), mWeights.end(), weights.begin(),
        [=] (float x) {return static_cast<int8_t>(roundf(std::max(std::min(x * mult, 127.0F), -127.0F))); });
        // we have FP16 linear output
        mOutActivationScale = 1.0F;
        CHECK_CUDA(cudaMemcpy(mDeviceWeights, weights.data(), weights.size() * sizeof(int8_t), cudaMemcpyHostToDevice));
    }
}

// conv3D1X1X1K4PluginCreator methods
conv3D1X1X1K4PluginCreator::conv3D1X1X1K4PluginCreator()
{
    mPluginAttributes.emplace_back(PluginField("inputChannels", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("weights", nullptr, PluginFieldType::kFLOAT32));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* conv3D1X1X1K4PluginCreator::getPluginName() const
{
    return CONV3D1X1X1K4_PLUGIN_NAME;
}

const char* conv3D1X1X1K4PluginCreator::getPluginVersion() const
{
    return CONV3D1X1X1K4_PLUGIN_VERSION;
}

const PluginFieldCollection* conv3D1X1X1K4PluginCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2DynamicExt* conv3D1X1X1K4PluginCreator::createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc)
{
    int inputChannels = 0;
    std::vector<float> weights;

    const PluginField* fields = fc->fields;
    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "inputChannels"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            inputChannels = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attrName, "weights"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            weights.resize(fields[i].length);
            memcpy(weights.data(), fields[i].data, fields[i].length * sizeof(float));
        }
    }

    conv3D1X1X1K4Plugin* obj = new conv3D1X1X1K4Plugin(inputChannels, weights);
    obj->setPluginNamespace(mNamespace.c_str());
    obj->initialize();
    return obj;
}

IPluginV2DynamicExt* conv3D1X1X1K4PluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    conv3D1X1X1K4Plugin* obj = new conv3D1X1X1K4Plugin{serialData, serialLength}; 
    obj->setPluginNamespace(mNamespace.c_str());
    obj->initialize();
    return obj;
}
