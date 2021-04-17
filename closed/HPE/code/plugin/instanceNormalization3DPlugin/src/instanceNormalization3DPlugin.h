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
#ifndef TRT_INSTANCE_NORMALIZATION_3D_PLUGIN_H
#define TRT_INSTANCE_NORMALIZATION_3D_PLUGIN_H
#include "serialize.hpp"
#include "plugin.h"
#include "instance_norm_fwd.h"
#include <cudnn.h>
#include <cuda_fp16.h>
#include <vector>
#include <iostream>
#include <string>

typedef unsigned short half_type;

namespace nvinfer1
{
namespace plugin
{
using namespace instance_norm_impl;
class InstanceNormalization3DPlugin final : public nvinfer1::IPluginV2DynamicExt
{

public:
  InstanceNormalization3DPlugin(float epsilon, nvinfer1::Weights const& scale, 
                                nvinfer1::Weights const& bias, int relu = 0, float alpha = 0.f);
  InstanceNormalization3DPlugin(float epsilon, const std::vector<float>& scale, 
                                const std::vector<float>& bias, int relu = 0, float alpha = 0.f);
  InstanceNormalization3DPlugin(void const* serialData, size_t serialLength);

  InstanceNormalization3DPlugin() = delete;

  ~InstanceNormalization3DPlugin() override;

  int getNbOutputs() const override;

  // DynamicExt plugins returns DimsExprs class instead of Dims
  using nvinfer1::IPluginV2::getOutputDimensions;
  DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) override;

  int initialize() override;

  void terminate() override;

  using nvinfer1::IPluginV2::getWorkspaceSize;
  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs, const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const override;

  using nvinfer1::IPluginV2::enqueue;
  int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
              const void* const* inputs, void* const* outputs,
              void* workspace,
              cudaStream_t stream) override;

  size_t getSerializationSize() const override;

  void serialize(void* buffer) const override;

  // DynamicExt plugin supportsFormat update.
  bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) override;

  const char* getPluginType() const override;

  const char* getPluginVersion() const override;

  void destroy() override;

  nvinfer1::IPluginV2DynamicExt* clone() const override;

  void setPluginNamespace(const char* pluginNamespace) override;

  const char* getPluginNamespace() const override;

  DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override;

  using nvinfer1::IPluginV2Ext::configurePlugin;
  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
                       const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) override;
private:
    float _epsilon;
    float _alpha;
    int _relu;
    int   _nchan;
    std::vector<float> _h_scale;
    std::vector<float> _h_bias;
    float* _d_scale;
    float* _d_bias;
    cudnnHandle_t _cudnn_handle;
    cudnnTensorDescriptor_t _x_desc, _y_desc, _b_desc;
    const char* mPluginNamespace;
    std::string mNamespace;
    bool initialized{false};

    // NDHWC implementation
    InstanceNormFwdParams _params;
    InstanceNormFwdContext _context;

    float _in_scale;
    float _out_scale;
};

class InstanceNormalization3DPluginCreator : public BaseCreator
{
public:
  InstanceNormalization3DPluginCreator();

  ~InstanceNormalization3DPluginCreator() override = default;

  const char* getPluginName() const override;

  const char* getPluginVersion() const override;

  const PluginFieldCollection* getFieldNames() override;

  IPluginV2DynamicExt* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) override;

  IPluginV2DynamicExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;

private:
  static PluginFieldCollection mFC;
  static std::vector<PluginField> mPluginAttributes;
  std::string mNamespace;
};
} //namespace plugin
} //namespace nvinfer1

#endif // TRT_INSTANCE_NORMALIZATION_3D_PLUGIN_H
