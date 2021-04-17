#ifndef TRT_SERIAL3_CONV2D_PLUGIN_H
#define TRT_SERIAL3_CONV2D_PLUGIN_H

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <algorithm>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>
#include <cassert>

using namespace nvinfer1::plugin;
namespace nvinfer1
{
namespace plugin
{
class Serial3Conv2dPlugin : public IPluginV2DynamicExt
{
private:
    std::string mNamespace;
    std::string mLayerName;

    int mDevice;
    int mSmCount;

    IGpuAllocator* mAllocator;

    bool mDataCopied;
    bool mInitialized;
    std::vector<std::vector<int8_t>> mWeights;
    std::vector<std::vector<float>> mBiases;
    std::vector<std::vector<float>> mScales;
    float mAddScale;
    std::vector<int> mOutlayout;
    
    
    std::vector<int8_t *> mDeviceWeights;
    std::vector<float *> mDeviceBiases;
    std::vector<float *> mDeviceScales;
public:

    Serial3Conv2dPlugin(
        const std::string layerName, 
        const std::vector<std::vector<int8_t>>& weights,
        const std::vector<std::vector<float>>& biases,
        const std::vector<std::vector<float>>& scales,
        const float scale_add,
        const std::vector<int>& layout);

    Serial3Conv2dPlugin(const std::string layerName, 
                        const void* data, 
                        size_t length
                        ); 

    Serial3Conv2dPlugin() = default;
    ~Serial3Conv2dPlugin() override;

    void setPluginNamespace(const char* libNamespace) override;
    const char* getPluginNamespace() const override;

    int initialize() override;
    void terminate() override;
    void destroy() override;

    void attachToContext(cudnnContext* cudnn, cublasContext* cublas, IGpuAllocator* allocator) override;
    void detachFromContext() override;

    int getNbOutputs() const override;  

    DimsExprs getOutputDimensions (int outputIndex, const DimsExprs *inputs, int nbInputs, IExprBuilder &exprBuilder) override; 

    virtual size_t getWorkspaceSize(const PluginTensorDesc *inputs, int nbInputs, const PluginTensorDesc *outputs, int nbOutputs) const override;

    virtual int enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) override;

    size_t getSerializationSize() const override;
    void serialize(void* buffer) const override;

    const char* getPluginType() const override;
    const char* getPluginVersion() const override;

    //IPluginV2Ext* clone() const override;
    IPluginV2DynamicExt * clone() const override;  

    // IPluginV2Ext methods
    DataType getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const override; 

    // IPluginV2IOExt methods
    void configurePlugin(const DynamicPluginTensorDesc *in, int nbInputs, const DynamicPluginTensorDesc *out, int nbOutputs) override;
    bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) override;
}; /* class Serial3Conv2dPlugin */

class Serial3Conv2dPluginCreator : public IPluginCreator
{
private:
    std::string mNamespace;
    static PluginFieldCollection mFieldNames;
    static std::vector<PluginField> mPluginAttributes;

public:
    Serial3Conv2dPluginCreator();
    const char* getPluginName() const override;
    const char* getPluginVersion() const override;

    const PluginFieldCollection* getFieldNames() override;

    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) override;
    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;

    void setPluginNamespace(const char* pluginNamespace) override;
    const char* getPluginNamespace() const override;
}; /* class Serial3Conv2dPluginCreator */

} /* namespace plugin */
}
#endif /* TRT_SERIAL3_CONV2D_PLUGIN_H */
