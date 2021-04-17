#include "Serial3Conv2dPlugin.h"
#include "helper.h"

#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <math.h>
#include <NvInferRuntimeCommon.h>
#include <exception>
#include "fuseconv3.h"
//#if CUDA_VERSION >= 10000 && INCLUDE_MMA_KERNELS

using namespace nvinfer1;

namespace
{
static const char* SERIAL3_CONV2D_PLUGIN_NAME{"Serial3Conv2d_TRT"};
static const char* SERIAL3_CONV2D_PLUGIN_VERSION{"1"};
}

REGISTER_TENSORRT_PLUGIN(Serial3Conv2dPluginCreator);

PluginFieldCollection Serial3Conv2dPluginCreator::mFieldNames{};
std::vector<PluginField> Serial3Conv2dPluginCreator::mPluginAttributes{};

Serial3Conv2dPlugin::Serial3Conv2dPlugin(
    const std::string layerName, 
    const std::vector<std::vector<int8_t>>& weights,
    const std::vector<std::vector<float>>& biases,
    const std::vector<std::vector<float>>& scales,
    const float scale_add,
    const std::vector<int>& layout
)
    : mLayerName{layerName}, 
    mInitialized(false),
    mWeights(weights),    
    mBiases(biases),
    mScales(scales),
    mAddScale(scale_add),
    mOutlayout(layout)
{
    mDevice = -1;
    mSmCount = -1;
    mDataCopied = false;
    initialize();

}

Serial3Conv2dPlugin::Serial3Conv2dPlugin(
    const std::string layerName, 
    const void* data
    , size_t length
    )
    : mLayerName(layerName),
    mInitialized(false)
{

    const char* d = reinterpret_cast<const char *>(data);

    mDevice = -1;
    mSmCount = -1;
    mDataCopied = false;

    mBiases.resize(1);
    int nSize; 
    nSize = read<int>(d);
    mBiases[0].resize(nSize);
    read(d, &mBiases[0].front(), mBiases[0].size());

    mWeights.resize(1);
    nSize = read<int>(d);
    mWeights[0].resize(nSize);
    read(d, &mWeights[0].front(), mWeights[0].size());

    mScales.resize(1);
    nSize = read<int>(d);
    mScales[0].resize(nSize);
    read(d, &mScales[0].front(), mScales[0].size());

    read(d, &mAddScale, 1);

    nSize = read<int>(d);
    mOutlayout.resize(nSize);
    read(d, &mOutlayout.front(), mOutlayout.size());
    
    assert(d == a + length);
}

Serial3Conv2dPlugin::~Serial3Conv2dPlugin() = default;


size_t Serial3Conv2dPlugin::getSerializationSize() const
{
    return 4*sizeof(int) + mOutlayout.size()*sizeof(int) + sizeof(float)
    + mScales[0].size()*sizeof(float)
    + mBiases[0].size() * sizeof(float)
        + mWeights[0].size() * sizeof(char);
}

void Serial3Conv2dPlugin::serialize(void* data) const  
{
    char *d = reinterpret_cast<char *>(data);

    write(d, static_cast<int>(mBiases[0].size()));
    write(d, &mBiases[0].front(), mBiases[0].size());

    write(d, static_cast<int>(mWeights[0].size()));
    write(d, &mWeights[0].front(), mWeights[0].size());
    
    write(d, static_cast<int>(mScales[0].size()));
    write(d, &mScales[0].front(), mScales[0].size());

    write(d, &mAddScale, 1);

    write(d, static_cast<int>(mOutlayout.size()));
    write(d, &mOutlayout.front(), mOutlayout.size());
}

void Serial3Conv2dPlugin::setPluginNamespace(const char* libNamespace)
{
    mNamespace = libNamespace;
}

const char* Serial3Conv2dPlugin::getPluginNamespace() const
{
    return mNamespace.c_str();
}

int Serial3Conv2dPlugin::initialize()
{

    if (!mInitialized)
    {
        mDeviceWeights.resize(mWeights.size());
        for(int i = 0; i < static_cast<int>(mWeights.size()); ++i)
        {   
            cudaError_t status;
            status = cudaMalloc(&mDeviceWeights[i], mWeights[i].size() * sizeof(int8_t));
            status = cudaMemcpy(mDeviceWeights[i], &mWeights[i].front(), mWeights[i].size() * sizeof(int8_t), cudaMemcpyHostToDevice);
        }

        mDeviceBiases.resize(mBiases.size());
        for(int i = 0; i < static_cast<int>(mBiases.size()); ++i)
        {
            cudaMalloc(&mDeviceBiases[i], mBiases[i].size() * sizeof(float));
            cudaMemcpy(mDeviceBiases[i], &mBiases[i].front(), mBiases[i].size() * sizeof(float), cudaMemcpyHostToDevice);
        }

        mDeviceScales.resize(mScales.size());
        for(int i = 0; i < static_cast<int>(mScales.size()); ++i)
        {
            cudaMalloc(&mDeviceScales[i], mScales[i].size() * sizeof(float));
            cudaMemcpy(mDeviceScales[i], &mScales[i].front(), mScales[i].size() * sizeof(float), cudaMemcpyHostToDevice);
        }

        mInitialized = true;
    }
    return 0;
    
}

void Serial3Conv2dPlugin::terminate()
{
    if (mInitialized)
    {
        for(auto devicePtr: mDeviceWeights)
            cudaFree(devicePtr);
        mDeviceWeights.clear();

        for(auto devicePtr: mDeviceBiases)
            cudaFree(devicePtr);
        mDeviceBiases.clear();

        for(auto devicePtr: mDeviceScales)
            cudaFree(devicePtr);
        mDeviceScales.clear();

        mInitialized = false;
    }
    
}

void Serial3Conv2dPlugin::attachToContext(cudnnContext* cudnn, cublasContext* cublas, IGpuAllocator* allocator)
{
    //std::cout << "Serial3Conv2dPlugin::attachToContext" << std::endl;
}

void Serial3Conv2dPlugin::detachFromContext()
{
    //std::cout << "Serial3Conv2dPlugin::detachFromContext" << cudaSuccess << std::endl;
}

int Serial3Conv2dPlugin::getNbOutputs() const
{
    //std::cout << "Serial3Conv2dPlugin::getNbOutputs" << std::endl;
    return 1;
}


DimsExprs Serial3Conv2dPlugin::getOutputDimensions (int outputIndex, const DimsExprs *inputs, int nbInputs, IExprBuilder &exprBuilder)
{

    DimsExprs outDims{4, {inputs[0].d[0], inputs[0].d[1], inputs[0].d[2], inputs[0].d[3]}};
    return outDims;
}

void Serial3Conv2dPlugin::destroy()
{
    //std::cout << "Serial3Conv2dPlugin::destroy" << std::endl;
    delete this;
}

size_t Serial3Conv2dPlugin::getWorkspaceSize(const PluginTensorDesc *inputs, int nbInputs, const PluginTensorDesc *outputs, int nbOutputs) const
{
    //std::cout << "Serial3Conv2dPlugin::getWorkspaceSize" << std::endl;
    return 0;
}


int Serial3Conv2dPlugin::enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, 
 void *workspace, cudaStream_t stream)
{
    //std::cout << "Serial3Conv2dPlugin::enqueue" << std::endl;
    int batchSize = inputDesc[0].dims.d[0];
    
    fuseconv1(
        batchSize,  //batch
        const_cast<int8_t*>((const int8_t *)(inputs[1])),
        const_cast<int8_t*>((const int8_t *)(inputs[0])), 
         mDeviceWeights[0],
        (int8_t*)(outputs[0]),
        mDeviceScales[0],
        mDeviceBiases[0],
        mAddScale,
        stream);
        
    //std::cout << "Serial3Conv2dPlugin::enqueue over" << std::endl;
    return 0;
}

const char* Serial3Conv2dPlugin::getPluginType() const
{
    //std::cout << "Serial3Conv2dPlugin::getPluginType" << std::endl;
    return SERIAL3_CONV2D_PLUGIN_NAME;
}

const char* Serial3Conv2dPlugin::getPluginVersion() const
{
    return SERIAL3_CONV2D_PLUGIN_VERSION;
}

IPluginV2DynamicExt * Serial3Conv2dPlugin::clone() const
{
    IPluginV2DynamicExt* clone;

    clone = new Serial3Conv2dPlugin(mNamespace
                                    ,mWeights
                                    ,mBiases
                                    ,mScales
                                    ,mAddScale
                                    ,mOutlayout);

    return clone;
}


DataType Serial3Conv2dPlugin::getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const
{
    //std::cout << "Serial3Conv2dPlugin::getOutputDataType" << std::endl;
    assert(index == 0);
    assert(nbInputs == 2);
    return DataType::kINT8;
}


// IPluginV2IOExt methods
void Serial3Conv2dPlugin::configurePlugin(const DynamicPluginTensorDesc *in, int nbInputs, const DynamicPluginTensorDesc *out, int nbOutputs)
{

}


bool Serial3Conv2dPlugin::supportsFormatCombination(
    int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs)
{
    switch (pos)
    {
        case 0:
            //std::cout << "pos 0 layout is TensorFormat::kCHW32" << std::endl;
            return ((inOut[pos].format == TensorFormat::kCHW32) && (inOut[pos].type == DataType::kINT8)); 
            break;
        case 1:
            //std::cout << "pos 1 layout is TensorFormat::kCHW32" << std::endl;
            return ((inOut[pos].format == TensorFormat::kCHW32) && (inOut[pos].type == DataType::kINT8));
            break;
        case 2:
            //std::cout << "pos 1 layout is TensorFormat::kCHW32" << std::endl;
            return ((inOut[pos].format == TensorFormat::kCHW32) && (inOut[pos].type == DataType::kINT8));
            break;
    }
    return false;

}

Serial3Conv2dPluginCreator::Serial3Conv2dPluginCreator()
{
    //std::cout << "Serial3Conv2dPluginCreator called" << std::endl;
	mPluginAttributes.emplace_back(PluginField("c_br2c_w", nullptr, PluginFieldType::kFLOAT32, 128*512));
	mPluginAttributes.emplace_back(PluginField("s_br2c_b", nullptr, PluginFieldType::kFLOAT32, 512));

    mPluginAttributes.emplace_back(PluginField("dynamic_ranges", nullptr, PluginFieldType::kFLOAT32, 3));

    mFieldNames.nbFields = mPluginAttributes.size();
    mFieldNames.fields = mPluginAttributes.data();

}

const char* Serial3Conv2dPluginCreator::getPluginName() const
{
    //std::cout << "Serial3Conv2dPluginCreator getPluginName" << std::endl;
    return SERIAL3_CONV2D_PLUGIN_NAME;
}

const char* Serial3Conv2dPluginCreator::getPluginVersion() const
{
    //std::cout << "Serial3Conv2dPluginCreator getPluginVersion" << std::endl;
    return SERIAL3_CONV2D_PLUGIN_VERSION;
}

const PluginFieldCollection* Serial3Conv2dPluginCreator::getFieldNames()
{
    //std::cout << "Serial3Conv2dPluginCreator::getFieldNames" << std::endl;
    return &mFieldNames;
}


IPluginV2* Serial3Conv2dPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    //std::cout << "createPlugin called" << std::endl;
    std::vector<std::vector<float>> weights(1);
    std::vector<std::vector<float>> biases(1);
    std::vector<std::vector<float>> scales(1);
    std::vector<std::vector<int8_t>> weights_int8(1);
    std::vector<float> dynamic_ranges(3);
    std::vector<int> layout(1);

    float br2cChannelMax[512];

    for (int i = 0; i < fc->nbFields; i++)
    {
        const PluginField* fields = fc->fields;
        const char* attrName = fields[i].name;
        std::cout << "Fc name:" << attrName << " size:" << fields[i].length << std::endl;

        if (!strcmp(attrName, "c_br2c_w"))
        {
            assert(fields[i].type == PluginFieldType::kFLOAT32);
            weights[0].resize(fields[i].length);
            memcpy(weights[0].data(), fields[i].data, fields[i].length * sizeof(float));
        }
        else if (!strcmp(attrName, "s_br2c_b"))
        {
            assert(fields[i].type == PluginFieldType::kFLOAT32);
            biases[0].resize(fields[i].length);
            memcpy(biases[0].data(), fields[i].data, fields[i].length * sizeof(float));
        }
        else if (!strcmp(attrName, "dynamic_ranges"))
        {
            assert(fields[i].type == PluginFieldType::kFLOAT32);
            memcpy(dynamic_ranges.data(), fields[i].data, fields[i].length * sizeof(float));
        }
        else
        {
            std::ostringstream stream;
            stream << "invalid plugin field name '" << attrName << "'" << stream.str().c_str()<<std::endl;
        }
    }
    /******************************quant weight********************************/
    for (int k = 0; k < 512; k++)
    {
        float max = fabs(static_cast<double>(weights[0][k * 128]));

        for (int c = 1; c < 128; c++)
        {
            float value = fabs(static_cast<double>(weights[0][k * 128 + c]));
            if (value > max)
            {
                max = value;
            }
        }
        br2cChannelMax[k] = max;
    }


    /******************************create weight and transform weight layout********************************/    
    for (int i = 0; i < 512; i += 1)
    {
        for (int j = 0; j < 128; j += 1)
        {
            weights[0][i*128+j] = weights[0][i*128 + j] * 127.0f / br2cChannelMax[i] ;
        }
    }

    weights_int8[0].resize(128*512);
    for (int n = 0; n < 4; n++) 
    {  //outer o
     for (int kK = 0; kK < 128; kK += 32) 
     {  //outer i
      for (int i = 0; i < 128; i++) 
      {  //inner o
       for (int j = 0; j < 32; j++) 
       {  //inner i
            weights_int8[0][n * 128 * 128 + kK * 128 + i * 32 + j] =(char) std::roundf(weights[0][n * 128 * 128 + i * 128 + (kK + j)]);
       }
      }
     }
    }
 
    /******************************create scales********************************/
    float scale_short_cut = dynamic_ranges[0];
    float scale_in = dynamic_ranges[1];
    float scale_out = dynamic_ranges[2];

    float scale_add = scale_short_cut / scale_out;
    //float scale_add = 1 / scale_out;
    
    for (uint i=0; i< biases[0].size(); i++)
    {
        scales[0].push_back(scale_in / scale_out * br2cChannelMax[i] / 127.0f); 
    }


    /******************************create biases********************************/
    // bias = br2b_b * 127.0 / scale;
    for (uint i=0; i< biases[0].size(); i++)
    {
        biases[0][i] = biases[0][i] * 127.0f / scale_out;
    }

    Serial3Conv2dPlugin* layer = new Serial3Conv2dPlugin(name, weights_int8, biases, scales, scale_add, layout);
    //std::cout << "createPlugin::out" << std::endl;
    return layer;
}

IPluginV2* Serial3Conv2dPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    //std::cout << "Serial3Conv2dPluginCreator::deserializePlugin" << std::endl;
    return new Serial3Conv2dPlugin(name, serialData, serialLength);
}

void Serial3Conv2dPluginCreator::setPluginNamespace(const char* pluginNamespace)
{
    //std::cout << "Serial3Conv2dPluginCreator::setPluginNamespace" << std::endl;
    mNamespace = pluginNamespace;
}

const char* Serial3Conv2dPluginCreator::getPluginNamespace() const
{
    //std::cout << "Serial3Conv2dPluginCreator::getPluginNamespace" << std::endl;
    return mNamespace.c_str();
}


//#endif /* CUDA_VERSION >= 10000 && INCLUDE_MMA_KERNELS */
