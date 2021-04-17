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

#include "pixelShuffle3DPlugin.h"

using namespace nvinfer1;
using nvinfer1::plugin::pixelShuffle3DPlugin;
using nvinfer1::plugin::pixelShuffle3DPluginCreator;

#define WARP_SIZE 32

#define CHECK_CUDA(call)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t status = call;                                                                                     \
        if (status != cudaSuccess)                                                                                     \
        {                                                                                                              \
            return status;                                                                                             \
        }                                                                                                              \
    } while (0)

#define CHECK_CUDNN(call)                                                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
        cudnnStatus_t status = call;                                                                                   \
        if (status != CUDNN_STATUS_SUCCESS)                                                                            \
        {                                                                                                              \
            return status;                                                                                             \
        }                                                                                                              \
    } while (0)


////////////////////////////////////////////////////////////////////////////////////////////////////

static inline int div_up(int m, int n) 
{
    return (m + n - 1) / n;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

template <int ELEMENTS_PER_WARP_LOAD>
using Copy_int8_t =
    typename std::conditional<ELEMENTS_PER_WARP_LOAD == 32, int8_t,
        typename std::conditional<ELEMENTS_PER_WARP_LOAD == 64, uint16_t,
            typename std::conditional<ELEMENTS_PER_WARP_LOAD == 128, int,
                typename std::conditional<ELEMENTS_PER_WARP_LOAD == 256, int2, int4
                >::type
            >::type
        >::type
    >::type;

template <typename T, int ELEMENTS_PER_WARP_LOAD>
using Copy_t = Copy_int8_t<sizeof(T) / sizeof(int8_t) * ELEMENTS_PER_WARP_LOAD>;

template<int ELEMENTS_PER_THREAD>
using copy_int8_t = Copy_t<int8_t, WARP_SIZE * ELEMENTS_PER_THREAD>;

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, int ELEMENTS_PER_THREAD>
union Access_t
{
    Copy_t<T, WARP_SIZE * ELEMENTS_PER_THREAD> v;
    T x[ELEMENTS_PER_THREAD]; 
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int ELEMENTS_PER_THREAD>
__global__ void copy2d_scaled(copy_int8_t<ELEMENTS_PER_THREAD> * __restrict out, int dst_pitch,
                              const copy_int8_t<ELEMENTS_PER_THREAD> * __restrict in, 
                              int src_pitch, int transfer_width, float scale) 
{
    typedef Access_t<int8_t, ELEMENTS_PER_THREAD> access_t;

    const int row_idx =blockIdx.x * blockDim.x + threadIdx.x;

    if (row_idx >= transfer_width / ELEMENTS_PER_THREAD) return; 

    const int in_idx = row_idx + src_pitch / ELEMENTS_PER_THREAD * blockIdx.y;
    const int out_idx = row_idx + dst_pitch / ELEMENTS_PER_THREAD * blockIdx.y;

    access_t elem;

    elem.v = in[in_idx];
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++)
    {
        elem.x[i] = __float_as_int(min(max(__int2float_rn(elem.x[i]) * scale + 12582912.0F, 12582785.0F), 12583039.0F));  
    }

    out[out_idx] = elem.v;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void copy2d_scaled_dispatch(int8_t * out, int dst_pitch, const int8_t * in, int src_pitch, 
                       int transfer_width, int transfer_height,
                       float scale, cudaStream_t stream) 
{
    const int block_sz = 256;

    if (transfer_width % 16 == 0)
    {
        dim3 grid = dim3(div_up(transfer_width / 16, block_sz), transfer_height);
        copy2d_scaled<16><<<grid, block_sz, 0, stream>>>(reinterpret_cast<int4 *>(out), dst_pitch, 
                                                     reinterpret_cast<const int4 *>(in), src_pitch, transfer_width, scale);
    }
    else if (transfer_width % 8 == 0)
    {
        dim3 grid = dim3(div_up(transfer_width / 8, block_sz), transfer_height);
        copy2d_scaled<8><<<grid, block_sz, 0, stream>>>(reinterpret_cast<int2 *>(out), dst_pitch, 
                                                     reinterpret_cast<const int2 *>(in), src_pitch, transfer_width, scale);
    }
    else if (transfer_width % 4 == 0)
    {
        dim3 grid = dim3(div_up(transfer_width / 4, block_sz), transfer_height);
        copy2d_scaled<4><<<grid, block_sz, 0, stream>>>(reinterpret_cast<int *>(out), dst_pitch, 
                                                     reinterpret_cast<const int *>(in), src_pitch, transfer_width, scale);
    }
    else
    {
        dim3 grid = dim3(div_up(transfer_width / 1, block_sz), transfer_height);
        copy2d_scaled<1><<<grid, block_sz, 0, stream>>>(reinterpret_cast<int8_t *>(out), dst_pitch, 
                                                     reinterpret_cast<const int8_t *>(in), src_pitch, transfer_width, scale);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// This is derived from: https://fgiesen.wordpress.com/2012/03/28/half-to-float-done-quic/
inline float half_to_float_fast(unsigned short value)
{
    union F32
    {
        unsigned int u;
        float f;
    };
    static const F32 magic = {(254 - 15) << 23};
    static const F32 was_infnan = {(127 + 16) << 23};
    F32 result;
    result.u = (value & 0x7fff) << 13; // exponent/mantissa bits
    result.f *= magic.f;               // exponent adjust
    if (result.f >= was_infnan.f)
    { // make sure Inf/NaN survive
        result.u |= 255 << 23;
    }
    result.u |= (value & 0x8000) << 16; // sign bit
    return result.f;
}

namespace 
{
    const char* PIXELSHUFFLE3D_PLUGIN_VERSION{"1"};
    const char* PIXELSHUFFLE3D_PLUGIN_NAME{"PIXELSHUFFLE3D_TRT"};
}

REGISTER_TENSORRT_PLUGIN(pixelShuffle3DPluginCreator);

PluginFieldCollection pixelShuffle3DPluginCreator::mFC{};
std::vector<PluginField> pixelShuffle3DPluginCreator::mPluginAttributes;

pixelShuffle3DPlugin::pixelShuffle3DPlugin(
    int r, int s, int t)
    : mR(r)
    , mS(s)
    , mT(t)
    , mInScale(-1.f)
    , mInConcatScale(-1.f)
    , mOutScale(-1.f)
    , mNbInputs(1)
{
}

pixelShuffle3DPlugin::pixelShuffle3DPlugin(void const* serialData, size_t serialLength)
{
    deserialize_value(&serialData, &serialLength, &mR);
    deserialize_value(&serialData, &serialLength, &mS);
    deserialize_value(&serialData, &serialLength, &mT);
    deserialize_value(&serialData, &serialLength, &mInScale);
    deserialize_value(&serialData, &serialLength, &mInConcatScale);
    deserialize_value(&serialData, &serialLength, &mOutScale);
}

pixelShuffle3DPlugin::~pixelShuffle3DPlugin()
{
    terminate();
}

// pixelShuffle3DPlugin returns one output.
int pixelShuffle3DPlugin::getNbOutputs() const
{
    return 1;
}

DimsExprs pixelShuffle3DPlugin::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder)
{
    nvinfer1::DimsExprs output(inputs[0]);

    output.d[0] = inputs[0].d[0];
    output.d[1] = exprBuilder.operation(DimensionOperation::kFLOOR_DIV,  *inputs[0].d[1], *exprBuilder.constant(mR * mS * mT));
    if (nbInputs == 2) {
        output.d[1] = exprBuilder.operation(DimensionOperation::kSUM, *inputs[1].d[1], *output.d[1]);
    }
    output.d[2] = exprBuilder.operation(DimensionOperation::kPROD, *inputs[0].d[2], *exprBuilder.constant(mR));
    output.d[3] = exprBuilder.operation(DimensionOperation::kPROD, *inputs[0].d[3], *exprBuilder.constant(mS));
    output.d[4] = exprBuilder.operation(DimensionOperation::kPROD, *inputs[0].d[4], *exprBuilder.constant(mT));

    mNbInputs = nbInputs;

    return output;
}

int pixelShuffle3DPlugin::initialize()
{
    if (!initialized)
    {
        // Get GPU memory size.
        size_t freeMemSize{0};
        CHECK_CUDA(cudaMemGetInfo(&freeMemSize, &mGpuMemSize));
    }
    initialized = true;
    return 0;
}

void pixelShuffle3DPlugin::terminate()
{
    if (initialized)
    {
    }
    initialized = false;
    return;
}

size_t pixelShuffle3DPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs, const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const 
{ 
    return 0;
}


int pixelShuffle3DPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream)
{
    ASSERT(initialized);

    if (inputDesc[0].format == nvinfer1::PluginFormat::kLINEAR || inputDesc[0].format == nvinfer1::PluginFormat::kCDHW32)
    {

        nvinfer1::Dims input_dims = inputDesc[0].dims;
        int n = input_dims.d[0];
        int c = input_dims.d[1];
        int d = input_dims.d[2];
        int h = input_dims.d[3];
        int w = input_dims.d[4];

        _params.o = d * mR;
        _params.p = h * mS;
        _params.q = w * mT;
        _params.k = c/mT/mR/mS;
        _params.n = n;
        _params.r = mR;
        _params.s = mS;
        _params.t = mT;
        _params.scale = mInScale / mOutScale;


        _params.gmem_src = const_cast<void *>(inputs[0]);
        _params.gmem_dst = outputs[0];

        int k = _params.k;
        int o = _params.o;
        int p = _params.p;
        int q = _params.q;

        _params.output_stride = k * o * p * q ;

        int n2 = 0, k2 = 0, o2 = 0, p2 = 0, q2 = 0;
        size_t dst_offset = 0, dst_pitch = 0, src_pitch = 0, transfer_width = 0, transfer_height = 0;
        int sizeof_datatype = (inputDesc[0].type == DataType::kINT8)? sizeof(int8_t) : sizeof(float);

        if (mNbInputs == 2)
        {
            // Will work for both kLINEAR ( NCDHW ) and kINT8 ( NC/32DHW32 )

            nvinfer1::Dims     input2nd_dims = inputDesc[1].dims;
            n2 = input2nd_dims.d[0];
            k2 = input2nd_dims.d[1];
            o2 = input2nd_dims.d[2];
            p2 = input2nd_dims.d[3];
            q2 = input2nd_dims.d[4];

            _params.output_stride += k2 * o2 * p2 * q2;

            dst_pitch = _params.output_stride * sizeof_datatype;
            dst_offset = k * o * p * q * sizeof_datatype;

            src_pitch =      k2 * o2 * p2 * q2 * sizeof_datatype;
            transfer_width = k2 * o2 * p2 * q2 * sizeof_datatype;
            transfer_height = n2;
        }

        if (inputDesc[0].format == nvinfer1::PluginFormat::kCDHW32)
        {
            assert(mOutScale != 0.f);
            int res = pixel_shuffle_ncdhw32_to_ncdhw32_dispatch(_params, stream);
        }
        else
        {
            int res = pixel_shuffle_ncdhw_to_ncdhw_dispatch(_params, stream);
        }

        if (mNbInputs == 2)
        {
            if (inputDesc[0].format == nvinfer1::PluginFormat::kCDHW32)
            {
                copy2d_scaled_dispatch((int8_t *)_params.gmem_dst + dst_offset, dst_pitch,
                    (int8_t *)inputs[1], src_pitch, transfer_width, transfer_height, 
                    mInConcatScale / mOutScale, stream);
            }
            else
            {
            CHECK_CUDA(cudaMemcpy2DAsync((char *)_params.gmem_dst + dst_offset, dst_pitch,
                inputs[1], src_pitch, transfer_width, transfer_height, 
                cudaMemcpyDeviceToDevice, stream));
            }
        }
    }
    else
    {
        ASSERT(false && "Unexpected input format");
    }

    return 0;
}

size_t pixelShuffle3DPlugin::getSerializationSize() const
{
    return (serialized_size(mR) +
            serialized_size(mS) +
            serialized_size(mT) +
            serialized_size(mInScale) +
            serialized_size(mInConcatScale) +
            serialized_size(mOutScale));
}

void pixelShuffle3DPlugin::serialize(void *buffer) const
{
    serialize_value(&buffer, mR);
    serialize_value(&buffer, mS);
    serialize_value(&buffer, mT);
    serialize_value(&buffer, mInScale);
    serialize_value(&buffer, mInConcatScale);
    serialize_value(&buffer, mOutScale);
}

bool pixelShuffle3DPlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs)
{
    ASSERT(inOut && pos < (nbInputs + nbOutputs));

    bool support_fp32_linear = (inOut[pos].type == nvinfer1::DataType::kFLOAT
        && inOut[pos].format == nvinfer1::PluginFormat::kLINEAR
        && inOut[pos].type == inOut[0].type
        && inOut[pos].format == inOut[0].format);

    bool support_int8_cdhw32 = (inOut[pos].type == nvinfer1::DataType::kINT8
        && inOut[pos].format == nvinfer1::PluginFormat::kCDHW32
        && inOut[pos].type == inOut[0].type
        && inOut[pos].format == inOut[0].format);

    // Turn off FP32 support if the GPU memory is less than or equal to 8GB since the
    // FP32 path would lead to out-of-memory error.
    return (mGpuMemSize > (8ULL << 30) && support_fp32_linear) || support_int8_cdhw32;
}

const char* pixelShuffle3DPlugin::getPluginType() const
{
    return PIXELSHUFFLE3D_PLUGIN_NAME;
}

const char* pixelShuffle3DPlugin::getPluginVersion() const
{
    return PIXELSHUFFLE3D_PLUGIN_VERSION;
}

void pixelShuffle3DPlugin::destroy()
{ 
    delete this;
}

IPluginV2DynamicExt* pixelShuffle3DPlugin::clone() const
{ 
    auto plugin = new pixelShuffle3DPlugin{mR, mS, mT};
    plugin->setPluginNamespace(mPluginNamespace);
    plugin->initialize();
    return plugin;
}

// Set plugin namespace
void pixelShuffle3DPlugin::setPluginNamespace(const char* pluginNamespace)
{
    mPluginNamespace = pluginNamespace;
}

const char* pixelShuffle3DPlugin::getPluginNamespace() const
{
    return mPluginNamespace;
}

nvinfer1::DataType pixelShuffle3DPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    ASSERT(inputTypes && nbInputs > 0 && index == 0);

    return nvinfer1::DataType::kFLOAT;
}

void pixelShuffle3DPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs)
{
    mNbInputs = nbInputs;
    mInScale = in[0].desc.scale;
    mOutScale = out[0].desc.scale;
    if (nbInputs == 2)
    {
        mInConcatScale = in[1].desc.scale;
    }
}

// pixelShuffle3DPluginCreator methods
pixelShuffle3DPluginCreator::pixelShuffle3DPluginCreator()
{
    mPluginAttributes.emplace_back(PluginField("R", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("S", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("T", nullptr, PluginFieldType::kINT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* pixelShuffle3DPluginCreator::getPluginName() const
{
    return PIXELSHUFFLE3D_PLUGIN_NAME;
}

const char* pixelShuffle3DPluginCreator::getPluginVersion() const
{
    return PIXELSHUFFLE3D_PLUGIN_VERSION;
}

const PluginFieldCollection* pixelShuffle3DPluginCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2DynamicExt* pixelShuffle3DPluginCreator::createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc)
{
    int r {};
    int s {};
    int t {};
    const PluginField* fields = fc->fields;
    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "R"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            r = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attrName, "S"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            s = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attrName, "T"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            t = *(static_cast<const int*>(fields[i].data));
        }
    }

    pixelShuffle3DPlugin* obj = new pixelShuffle3DPlugin(r, s, t);
    obj->setPluginNamespace(mNamespace.c_str());
    obj->initialize();
    return obj;
}

IPluginV2DynamicExt* pixelShuffle3DPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    pixelShuffle3DPlugin* obj = new pixelShuffle3DPlugin{serialData, serialLength}; 
    obj->setPluginNamespace(mNamespace.c_str());
    obj->initialize();
    return obj;
}
