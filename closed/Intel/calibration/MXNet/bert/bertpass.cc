/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Copyright (c) 2019 by Contributors
 * \file subgraph_lib.cc
 * \brief subgraph operator implementation library file
 */

#include <math.h>
#include <iostream>
#include <algorithm>
#include <unordered_set>
#include <functional>
#include "mxnet/lib_api.h"
#include "mshadow/base.h"
#include <string>

using namespace mxnet::ext;


MXReturnValue custom_pass(mxnet::ext::Graph *g,
                     const std::unordered_map<std::string, std::string>& options) {

  auto nodes = g->topological_sort();
  //////////////// MHA remove reshapes & concat ///////////////////
  // find shape of weight / bias, number of heads, and count number of MHA layers
  std::string query0_weight = "bertencoder0_transformer0_dotproductselfattentioncell0_query_weight";
  std::string mult_qk0 = "bertencoder0_transformer0_dotproductselfattentioncell0_interleaved_matmul_selfatt_qk0";
  std::string str_projection = "_dotproductselfattentioncell0_fullyconnected0";
  int num_mha_layers = 0;
  int num_heads = 0;
  int head_dimension = 0;
  int shape0, shape1;
  for(Node* n : nodes){
      if (n->name.find(query0_weight) != std::string::npos) {
          std::string shape = n->attrs["__shape__"];
          int pos_comma = shape.find(",");
          shape0 = stoi(shape.substr(1, pos_comma-1));
          shape1 = stoi(shape.substr(pos_comma+2, shape.length()-pos_comma-3)); 
      }
      if (n->name.find(mult_qk0) != std::string::npos) {
          std::string h = n->attrs["heads"];
          num_heads = stoi(h);
      }
      if (n->name.find(str_projection) != std::string::npos) {
          num_mha_layers++;
      }
  }
  head_dimension = shape0 / num_heads;



  for(Node* n : nodes){

      // remove copy ops
      if (n->op.find("_contrib_interleaved_matmul_selfatt_valatt") != std::string::npos) {
          auto copynode = n->inputs[1].node;
          if(copynode->op.find("_copy") != std::string::npos) {
            n->inputs[1] = copynode->inputs[0];
          }
      }

      // find projection nodes and set new interleaved intputs
      if (n->name.find("_dotproductselfattentioncell0_fullyconnected0") != std::string::npos) {
          Node* node_projection = n;
          std::size_t pos = node_projection->name.find("_fullyconnected0");
          std::string base_name = n->name.substr(0,pos);

          //////////////////// WEIGHTS ////////////////////
          // create new arg with interleaved weights
          std::string name_qkv_weights_interleaved = base_name + "_qkv_interleaved_weight";

          
          // create a new input Node
          Node* node_qkv_weights = g->addNode(name_qkv_weights_interleaved, "null");
          node_qkv_weights->alloc_arg({3*shape0,shape1}, MXContext::CPU(0), kFloat32);

          MXTensor* qkv_weights_interleaved = node_qkv_weights->tensor;
          float* qkv_w_data = qkv_weights_interleaved->data<float>();
          // read from previous values and interleave them
          Node *query_w_node, *key_w_node, *value_w_node;
          for(Node* n : nodes){
            if(n->name == base_name+"_query_weight"){
              query_w_node = n;
            }
            if(n->name == base_name+"_key_weight"){
              key_w_node = n;
            }
            if(n->name == base_name+"_value_weight"){
              value_w_node = n;
            }
          }

          float* query_w_data = query_w_node->tensor->data<float>();
          float* key_w_data = key_w_node->tensor->data<float>();
          float* value_w_data = value_w_node->tensor->data<float>();
          for(int h=0; h<num_heads; ++h){
              for(int e=0; e<head_dimension*shape1; ++e){
                  qkv_w_data[h*head_dimension*shape1*3 + e] =
                      query_w_data[h*head_dimension*shape1 + e];
              }
              for(int e=0; e<head_dimension*shape1; ++e){
                  qkv_w_data[h*head_dimension*shape1*3 + head_dimension*shape1 + e] =
                      key_w_data[h*head_dimension*shape1 + e];
              }
              for(int e=0; e<head_dimension*shape1; ++e){
                  qkv_w_data[h*head_dimension*shape1*3 + 2*head_dimension*shape1 + e] =
                      value_w_data[h*head_dimension*shape1 + e];
              }
          }

          // set connection with new input
          node_projection->inputs[1].node = node_qkv_weights;
          node_projection->inputs[1].entry = 0;

          //////////////////// BIAS ////////////////////
          // create new arg with all bias
          std::string name_qkv_bias = base_name + "_qkv_bias";
          Node* node_qkv_bias = g->addNode(name_qkv_bias, "null");
          node_qkv_bias->alloc_arg({3*shape0,}, MXContext::CPU(0), kFloat32);
          float* qkv_bias_data = node_qkv_bias->tensor->data<float>();
          // read from previous values and join them
          Node *query_bias_node, *key_bias_node, *value_bias_node;
          for(Node* n : nodes){
            if(n->name == base_name+"_query_bias"){
              query_bias_node = n;
            }
            if(n->name == base_name+"_key_bias"){
              key_bias_node = n;
            }
            if(n->name == base_name+"_value_bias"){
              value_bias_node = n;
            }
          }
          float* query_bias_data = query_bias_node->tensor->data<float>();
          float* key_bias_data = key_bias_node->tensor->data<float>();
          float* value_bias_data = value_bias_node->tensor->data<float>();
          int counter = 0;
          for(int e=0; e < num_heads*3; e+=3){
              for(int h=e*head_dimension; h < e*head_dimension + head_dimension; h++) {
                qkv_bias_data[h] = query_bias_data[counter++];
              }
          }
          counter = 0;
          for(int e=1; e < num_heads*3; e+=3){
              for(int h=e*head_dimension; h < e*head_dimension + head_dimension; h++) {
                qkv_bias_data[h] = key_bias_data[counter++];
              }
          }
          counter = 0;
          for(int e=2; e < num_heads*3; e+=3){
              for(int h=e*head_dimension; h < e*head_dimension + head_dimension; h++) {
                qkv_bias_data[h] = value_bias_data[counter++];
              }
          }

          // set connection with new input
          node_projection->inputs[2].node = node_qkv_bias;
          node_projection->inputs[2].entry = 0;
      }
  }
  //////////////////////////////////////////////////////////////////

  return MX_SUCCESS;
}



static const float kUint8Range = 255.5;
static const float kInt8Range = 127.5;
static const size_t kInt32Range = 0x7fffffff;


template<typename T>
MSHADOW_XINLINE float MaxAbs(T a, T b) {
  return std::max(std::abs(static_cast<float>(a)), std::abs(static_cast<float>(b)));
}

static inline float GetQuantizeScale(const int dtype, const float data_min, const float data_max) {
  const float real_data_range = MaxAbs(data_min, data_max);
  const auto quantized_data_range = (dtype == mshadow::kInt8) ? kInt8Range : kUint8Range;
  // If real_data_range == 0, to avoid `inf` in scale, use a large number here, which is MAX_INT.
  return real_data_range ? quantized_data_range / real_data_range : mshadow::red::limits::MaxValue<int32_t>();
}

static inline mshadow::TypeFlag  GetFCInputDtype(Node* fcnode) {
  auto out_type = mshadow::kInt8;
  auto in_node = fcnode->inputs[0].node;

  if(in_node->attrs.count("shifted") || in_node->attrs.count("shifted_output")) {
    return mshadow::kInt8;
  }

  if(in_node->op.find("_contrib_quantize_v2") != std::string::npos) {
    if (in_node->attrs["out_type"].find("auto") != std::string::npos) {
      if (in_node->attrs.count("min_calib_range") && in_node->attrs.count("max_calib_range")) {
        if (std::stof(in_node->attrs["min_calib_range"]) >= 0.0) {
          out_type = mshadow::kUint8;
        } else {
          out_type = mshadow::kInt8;
        }
      }
    } else if (in_node->attrs["out_type"].compare("int8") == 0) { //equal
      out_type = mshadow::kInt8;
    } else if (in_node->attrs["out_type"].compare("uint8") == 0) {
      out_type = mshadow::kUint8;
    } 
  }

  return out_type;
}


MXReturnValue bias_to_s32(mxnet::ext::Graph *g,
                     const std::unordered_map<std::string, std::string>& options) {

  std::string fc_op = "_sg_mkldnn_fully_connected";

  auto HasFCBias = [&](Node* mkldnnfc) {
      if(mkldnnfc->subgraphs.size() < 1)
        return false;

      bool _hasbias = false;
      auto subgraph = mkldnnfc->subgraphs[0];
      auto sg_nodes = subgraph->topological_sort();
      for(Node* n : sg_nodes) {
        if(n->op.find("FullyConnected") != std::string::npos) {
          if(n->attrs["no_bias"].find("False") != std::string::npos) {
            _hasbias = true;
            return _hasbias;
          }
        }
      }
      return _hasbias;
  };


  auto nodes = g->topological_sort();
  for(Node* n : nodes){

      if (n->op.find(fc_op) != std::string::npos) {
        bool is_quantized = (n->attrs["quantized"].find("True") != std::string::npos);

        if(HasFCBias(n) && is_quantized) {
          auto input_node = n->inputs[0].node;

          float min_data = 0.0f;
          float max_data = 0.0f;
          if (input_node->attrs.count("min_calib_range") && input_node->attrs.count("max_calib_range")) {
            min_data = std::stof(input_node->attrs["min_calib_range"]);
            max_data = std::stof(input_node->attrs["max_calib_range"]);
          } else {
              continue;
          }
          auto bias_node = n->inputs[2].node;
          MXTensor* bias_tensor = bias_node->tensor;
          int8_t *bias_ptr = bias_tensor->data<int8_t>();
          size_t bias_size = bias_tensor->size();

          auto weight_node = n->inputs[1].node;
          MXTensor* weight_tensor = weight_node->tensor;
          int8_t *weight_ptr = weight_tensor->data<int8_t>();
          size_t weight_size = weight_tensor->size();

          int minmax_index = 5;
          auto ws = n->attrs.find("with_sum");
          if (ws != n->attrs.end()){
              if((n->attrs["with_sum"].find("True") != std::string::npos)){
                minmax_index++;
              }
          }

          const int min_w_index = minmax_index++;
          const int max_w_index = minmax_index++;
          const int min_b_index = minmax_index++;
          const int max_b_index = minmax_index++;
          float min_weight = n->inputs[min_w_index].node->tensor->data<float>()[0];
          float max_weight = n->inputs[max_w_index].node->tensor->data<float>()[0];
          float min_bias = n->inputs[min_b_index].node->tensor->data<float>()[0];
          float max_bias = n->inputs[max_b_index].node->tensor->data<float>()[0];


          auto input_dtype = GetFCInputDtype(n);

          float data_scale_ = GetQuantizeScale(input_dtype, min_data, max_data);
          float weight_scales = GetQuantizeScale(kInt8, min_weight, max_weight);

          float bias_scale = GetQuantizeScale(kInt8, min_bias, max_bias);
          float bias_int32_rescale = data_scale_ * weight_scales / bias_scale;

          float bias_max_rescale =
              mshadow::red::limits::MaxValue<int32_t>() / 2 / MaxAbs(min_bias, max_bias) / bias_scale;
          if (bias_int32_rescale > bias_max_rescale) {
            // avoid overflow on bias
            bias_int32_rescale = bias_max_rescale;
            float weight_rescale =
              bias_int32_rescale * bias_scale / data_scale_ / weight_scales;
            
            for (int32_t i = 0; i < static_cast<int32_t>(weight_size); ++i) {
              weight_ptr[i] = std::round(weight_ptr[i] * weight_rescale);
            }
            weight_scales *= weight_rescale;
            *(n->inputs[5].node->tensor->data<float>()) *= weight_rescale;
            *(n->inputs[6].node->tensor->data<float>()) *= weight_rescale;
          }
          Node* int32_bias_node = g->addNode(bias_node->name + "_s32", "null");
          int32_bias_node->alloc_arg(bias_tensor->shape, MXContext::CPU(0), kInt32);

          int32_t *s32_bias_ptr = int32_bias_node->tensor->data<int32_t>();
          
          for (int32_t i = 0; i < static_cast<int32_t>(bias_size); ++i) {
            s32_bias_ptr[i] = std::round(bias_ptr[i] * bias_int32_rescale);
          }

          if(n->attrs.count("shift_value")) {
              // (M, K) * (K, N) = (M, N)
              int32_t shift_value = std::stoi(n->attrs["shift_value"]);
              size_t M = weight_tensor->shape[0];
              size_t K = weight_tensor->shape[1];

              int32_t* shift_matrix = new int32_t[M];
              for(int i=0; i < M; i++) {
                    shift_matrix[i] = 0;
                    for(int j=0; j < K; j++) {
                        int i_j_index = i * K + j;
                        shift_matrix[i] +=  shift_value * weight_ptr[i_j_index];
                    }
              }

              for (int32_t i = 0; i < static_cast<int32_t>(bias_size); ++i) {
                s32_bias_ptr[i] -= shift_matrix[i];
              }
          }

          n->inputs[2].node = int32_bias_node;
          n->inputs[2].entry = 0;
        }
      }
  }

  return MX_SUCCESS;
}


MXReturnValue softmax_mask(mxnet::ext::Graph *g,
                           const std::unordered_map<std::string, std::string>& options) {
    
  auto nodes = g->topological_sort();
  std::string selfattn_node = "quantized__sg_mkldnn_contrib_interleaved_matmul_selfatt_qk_0";
  int num_heads = 0;
  for(Node* n : nodes){
      if (n->name.find(selfattn_node) != std::string::npos) {
           std::string h = n->attrs["heads"];
           num_heads = std::stoi(h);
      }
  }

  Node* sequence_final;
  for(Node* n : nodes) {
     if (n->name.find("bertencoder0_broadcast_add0") != std::string::npos) {

         auto onesMask = g->addNode("ones_mask", "ones_like");
         onesMask->inputs.resize(1);
         onesMask->inputs[0].node = n;
         onesMask->inputs[0].entry = 0;

         auto sequence = g->addNode("sequence_mask", "SequenceMask");
         sequence->inputs.resize(2);
         sequence->inputs[0].node = onesMask;
         sequence->inputs[0].entry = 0;
         sequence->inputs[1] = n->inputs[0].node->inputs[0];
         sequence->attrs["axis"] = "1";
         sequence->attrs["use_sequence_length"] = "True";
         sequence->attrs["value"] = "-1e12";

         auto mask_minus_one = g->addNode("minus_one_mask", "_minus_scalar");
         mask_minus_one->inputs.resize(1);
         mask_minus_one->inputs[0].node = sequence;
         mask_minus_one->inputs[0].entry = 0;
         mask_minus_one->attrs["scalar"] = "1";

         auto expdimsmask = g->addNode("exp_dims_mask", "expand_dims");
         expdimsmask->inputs.resize(1);
         expdimsmask->inputs[0].node = mask_minus_one;
         expdimsmask->inputs[0].entry = 0;
         expdimsmask->attrs["axis"] = "1";

         auto brdcsaxiss = g->addNode("broadcast_axis_mask", "broadcast_like");
         brdcsaxiss->inputs.resize(2);
         brdcsaxiss->inputs[0].node = expdimsmask;
         brdcsaxiss->inputs[0].entry = 0;
         brdcsaxiss->inputs[1].node = expdimsmask;
         brdcsaxiss->inputs[1].entry = 0;
         brdcsaxiss->attrs["lhs_axes"] = "1";
         brdcsaxiss->attrs["rhs_axes"] = "-1";

         auto head_axis = g->addNode("head_axis_mask", "expand_dims");
         head_axis->inputs.resize(1);
         head_axis->inputs[0].node = brdcsaxiss;
         head_axis->inputs[0].entry = 0;
         head_axis->attrs["axis"] = "1";

         auto broadcast_head = g->addNode("broadcast_head_mask", "broadcast_axis");
         broadcast_head->inputs.resize(1);
         broadcast_head->inputs[0].node = head_axis;
         broadcast_head->inputs[0].entry = 0;
         broadcast_head->attrs["axis"] = "1";
         broadcast_head->attrs["size"] = std::to_string(num_heads);

         auto bs_mul_head_shape = g->addNode("bs_mul_head_shape_mask", "reshape");
         bs_mul_head_shape->inputs.resize(1);
         bs_mul_head_shape->inputs[0].node = broadcast_head;
         bs_mul_head_shape->inputs[0].entry = 0;
         bs_mul_head_shape->attrs["shape"] = "(-1, 0, 0)";
         bs_mul_head_shape->attrs["reverse"] = "True";

         sequence_final = bs_mul_head_shape;
     }
   }

  // find softmax nodes and set new interleaved intputs
  int cnt = 0;
  for(Node* n : nodes){

    if (n->op.find("softmax") != std::string::npos) {
       auto matmul_qk = n->inputs[0].node;
       NodeEntry seq_entry = {sequence_final, 0};
       matmul_qk->inputs.push_back(seq_entry);
       matmul_qk->attrs["with_mask"] = "True";
       n->attrs["use_length"] = "False";
       n->inputs.pop_back();
   }
  }

  return MX_SUCCESS;

}



MXReturnValue normal_softmax(mxnet::ext::Graph *g,
                             const std::unordered_map<std::string, std::string>& options) {

  auto nodes = g->topological_sort();

  // find softmax nodes and disable use_length attribute
  for(Node* n : nodes){
    if (n->op.find("softmax") != std::string::npos) {
       n->attrs["use_length"] = "False";
       n->inputs.pop_back();
   }
  }

  return MX_SUCCESS;

}

REGISTER_PASS(custom_pass)
.setBody(custom_pass);

REGISTER_PASS(bias_to_s32)
.setBody(bias_to_s32);

REGISTER_PASS(softmax_mask)
.setBody(softmax_mask);

REGISTER_PASS(normal_softmax)
.setBody(normal_softmax);

MXReturnValue initialize(int version) {
  if (version >= 10400) {
    std::cout << "MXNet version " << version << " supported" << std::endl;
    return MX_SUCCESS;
  } else {
    std::cout << "MXNet version " << version << " not supported" << std::endl;
    return MX_FAIL;
  }
}
