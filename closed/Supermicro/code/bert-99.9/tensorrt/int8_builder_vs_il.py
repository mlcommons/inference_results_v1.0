# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import onnx
import tensorrt as trt
import json

from code.bert.tensorrt.builder_utils import add_gelu, mark


def bert_encoder_layer_int8_vs_il(cfg, max_seqlen, weights_dict, network, input_tensor, cu_seqlens, layer):
    """Builds one encoder layer in INT8 mode with var seqlen and using interleaved layout.
    Sets the dynamic ranges extracted from the qat checkpoint."""

    plg_registry = trt.get_plugin_registry()
    qkv_plg_creator = plg_registry.get_plugin_creator("CustomQKVToContextPluginDynamic", "3", "")
    pc_skln = plg_registry.get_plugin_creator("CustomSkipLayerNormPluginDynamic", "3", "")
    dtype = trt.int8
    N = cfg.N
    H = cfg.H
    prefix = 'l{}_'.format(layer)

    dr_input = weights_dict[prefix + 'attention_self_query_input_amax']
    assert(dr_input == weights_dict[prefix + 'attention_self_key_input_amax'])
    assert(dr_input == weights_dict[prefix + 'attention_self_value_input_amax'])
    input_tensor.set_dynamic_range(-dr_input, dr_input)
    # FC QKV
    dr_qkv = max(
        weights_dict[prefix + 'attention_self_qv_a_input_quantizer_amax'],
        weights_dict[prefix + 'attention_self_qv_b_input_quantizer_amax'],
        weights_dict[prefix + 'attention_self_av_b_input_quantizer_amax'],
    )
    Wqkv = np.zeros((3, cfg.hidden_size, cfg.hidden_size), np.float32)
    Bqkv = np.zeros((3, cfg.hidden_size), np.float32)
    Wqkv[0, :, :] = weights_dict[prefix + 'attention_self_query_kernel']
    Wqkv[1, :, :] = weights_dict[prefix + 'attention_self_key_kernel']
    Wqkv[2, :, :] = weights_dict[prefix + 'attention_self_value_kernel']
    Bqkv[0, :] = weights_dict[prefix + 'attention_self_query_bias']
    Bqkv[1, :] = weights_dict[prefix + 'attention_self_key_bias']
    Bqkv[2, :] = weights_dict[prefix + 'attention_self_value_bias']

    Wqkv = np.ascontiguousarray(Wqkv.reshape((3, N, H, N, H)))
    Bqkv = np.ascontiguousarray(Bqkv.reshape((3, N, H)))

    fc_qkv = network.add_convolution(input_tensor, cfg.qkv_size, (1, 1), Wqkv, Bqkv)
    fc_qkv.name = prefix + 'fc_qkv'
    fc_qkv_out = fc_qkv.get_output(0)
    fc_qkv_out.name = prefix + 'attention_self_qkv_mult'
    fc_qkv_out.set_dynamic_range(-dr_qkv, dr_qkv)
    # QKV2CTX
    dr_probs = weights_dict[prefix + 'attention_self_av_a_input_quantizer_amax']
    dq_probs = dr_probs / 127.0
    pf_hidden_size = trt.PluginField("hidden_size", np.array([cfg.hidden_size], np.int32), trt.PluginFieldType.INT32)
    pf_num_heads = trt.PluginField("num_heads", np.array([cfg.N], np.int32), trt.PluginFieldType.INT32)
    pf_dq_probs = trt.PluginField("dq_probs", np.array([dq_probs], np.float32), trt.PluginFieldType.FLOAT32)

    pfc = trt.PluginFieldCollection([pf_hidden_size, pf_num_heads, pf_dq_probs])
    qkv2ctx_plug = qkv_plg_creator.create_plugin("qkv2ctx", pfc)

    dr_ctx = weights_dict[prefix + 'attention_output_dense_input_amax']
    qkv2ctx_layer = network.add_plugin_v2([fc_qkv_out, cu_seqlens, max_seqlen], qkv2ctx_plug)
    qkv2ctx_layer.name = prefix + 'qkv_to_ctx'
    qkv2ctx_out = qkv2ctx_layer.get_output(0)
    qkv2ctx_out.set_dynamic_range(-dr_ctx, dr_ctx)
    # FC AOUT
    dr_fc_aout = weights_dict[prefix + 'attention_output_add_local_input_quantizer_amax']
    Waout = weights_dict[prefix + 'attention_output_dense_kernel']
    Baout = weights_dict[prefix + 'attention_output_dense_bias']
    fc_aout = network.add_convolution(qkv2ctx_out, cfg.hidden_size, (1, 1), Waout, Baout)
    fc_aout.precision = dtype
    fc_aout.name = prefix + 'fc_aout'
    fc_aout_out = fc_aout.get_output(0)
    fc_aout_out.dtype = dtype
    fc_aout_out.set_dynamic_range(-dr_fc_aout, dr_fc_aout)
    # Skip-Layernorm 1
    dr_skln1 = weights_dict[prefix + 'intermediate_dense_input_amax']
    pf_ld = trt.PluginField("ld", np.array([cfg.hidden_size], np.int32), trt.PluginFieldType.INT32)
    pf_beta = trt.PluginField("beta", weights_dict[prefix + 'attention_output_layernorm_beta'], trt.PluginFieldType.FLOAT32)
    pf_gamma = trt.PluginField("gamma", weights_dict[prefix + 'attention_output_layernorm_gamma'], trt.PluginFieldType.FLOAT32)
    fields = [pf_beta, pf_gamma]
    pfc = trt.PluginFieldCollection(fields)
    skipln_plug = pc_skln.create_plugin("skipln", pfc)

    fc_aout_out.dtype = dtype

    skipln_inputs = [fc_aout_out, input_tensor]
    skln1 = network.add_plugin_v2(skipln_inputs, skipln_plug)
    skln1.name = prefix + 'skln_1'
    skln1_out = skln1.get_output(0)
    skln1_out.dtype = dtype
    skln1_out.set_dynamic_range(-dr_skln1, dr_skln1)
    # FC MID
    Wmid = weights_dict[prefix + 'intermediate_dense_kernel']
    Bmid = weights_dict[prefix + 'intermediate_dense_bias']
    fc_mid = network.add_convolution(skln1_out, cfg.mid_size, (1, 1), Wmid, Bmid)
    fc_mid.name = prefix + 'fc_mid'
    fc_mid_out = fc_mid.get_output(0)
    # GELU
    dr_gelu = weights_dict[prefix + 'output_dense_input_amax']
    gelu_layer = add_gelu(network, fc_mid_out)
    gelu_layer.name = prefix + 'gelu'
    gelu_out = gelu_layer.get_output(0)
    gelu_out.set_dynamic_range(-dr_gelu, dr_gelu)
    # FC OUT
    dr_fc_out = weights_dict[prefix + 'output_add_local_input_quantizer_amax']
    Wout = weights_dict[prefix + 'output_dense_kernel']
    Bout = weights_dict[prefix + 'output_dense_bias']
    fc_out = network.add_convolution(gelu_out, cfg.hidden_size, (1, 1), Wout, Bout)
    fc_out.name = prefix + 'fc_out'
    fc_out.precision = dtype
    fc_out_out = fc_out.get_output(0)
    fc_out_out.dtype = dtype
    fc_out_out.set_dynamic_range(-dr_fc_out, dr_fc_out)
    # Skip-Layernorm 2
    pf_beta = trt.PluginField("beta", weights_dict[prefix + 'output_layernorm_beta'], trt.PluginFieldType.FLOAT32)
    pf_gamma = trt.PluginField("gamma", weights_dict[prefix + 'output_layernorm_gamma'], trt.PluginFieldType.FLOAT32)
    fields = [pf_beta, pf_gamma]
    pfc = trt.PluginFieldCollection(fields)
    skipln_plug = pc_skln.create_plugin("skipln", pfc)

    skln1_out.dtype = dtype  # It does not build without setting this here, in addition to above. WHY??!?!

    skipln_inputs = [fc_out_out, skln1_out]
    skln2 = network.add_plugin_v2(skipln_inputs, skipln_plug)
    skln2.name = prefix + 'skln_2'
    skln2_out = skln2.get_output(0)

    return skln2_out


def bert_squad_int8_vs_il(network, weights_dict, cfg, input_shape, cu_seqlens_shape):
    """Create BERT network with INT8, var seqlen and using interleaved layout."""

    # Instantiate all the plugins
    plg_registry = trt.get_plugin_registry()

    pc_emb = plg_registry.get_plugin_creator("CustomEmbLayerNormPluginDynamic", "2", "")

    wbeta = trt.PluginField("bert_embeddings_layernorm_beta", weights_dict["bert_embeddings_layernorm_beta"], trt.PluginFieldType.FLOAT32)
    wgamma = trt.PluginField("bert_embeddings_layernorm_gamma", weights_dict["bert_embeddings_layernorm_gamma"], trt.PluginFieldType.FLOAT32)
    wwordemb = trt.PluginField("bert_embeddings_word_embeddings", weights_dict["bert_embeddings_word_embeddings"], trt.PluginFieldType.FLOAT32)
    wtokemb = trt.PluginField("bert_embeddings_token_type_embeddings", weights_dict["bert_embeddings_token_type_embeddings"], trt.PluginFieldType.FLOAT32)
    wposemb = trt.PluginField("bert_embeddings_position_embeddings", weights_dict["bert_embeddings_position_embeddings"], trt.PluginFieldType.FLOAT32)

    output_fp16 = trt.PluginField("output_fp16", np.array([int(trt.float16)]).astype(np.int32), trt.PluginFieldType.INT32)

    pfc = trt.PluginFieldCollection([wbeta, wgamma, wwordemb, wtokemb, wposemb, output_fp16])
    embln_plugin = pc_emb.create_plugin("embeddings", pfc)

    dtype = trt.int8

    input_ids = network.add_input(name="input_ids", dtype=trt.int32, shape=input_shape)
    segment_ids = network.add_input(name="segment_ids", dtype=trt.int32, shape=input_shape)

    cu_seqlens = network.add_input(name="cu_seqlens", dtype=trt.int32, shape=cu_seqlens_shape)

    # Dummy input used to indicate maximum sequence length to plugins
    max_seqlen = network.add_input(name="max_seqlen", dtype=trt.int32, shape=(-1,))

    inputs = [input_ids, segment_ids, cu_seqlens, max_seqlen]
    emb_layer = network.add_plugin_v2(inputs, embln_plugin)
    emb_layer.name = 'embln'

    embeddings = emb_layer.get_output(0)

    # We ideally want to go to int8 before the shuffle
    dr_emb = weights_dict['l0_attention_self_query_input_amax']
    embeddings.dtype = dtype
    embeddings.set_dynamic_range(-dr_emb, dr_emb)

    shuffle = network.add_shuffle(embeddings)
    print(embeddings.shape)
    shuffle.second_transpose = (2, 1, 0, 3)
    shuffle_out = shuffle.get_output(0)
    print(shuffle.get_output(0).shape)
    # shuffle_out.set_dynamic_range(-dr_emb, dr_emb)
    shuffle_out.dtype = dtype
    shuffle_out.allowed_formats = 1 << int(trt.TensorFormat.CHW32)

    embeddings = shuffle_out

    layer = 0
    for layer in range(cfg.L):
        embeddings = bert_encoder_layer_int8_vs_il(cfg, max_seqlen, weights_dict, network, embeddings, cu_seqlens, layer)

    Wsquad = weights_dict['cls_squad_output_weights']
    Bsquad = weights_dict['cls_squad_output_bias']

    dr_out = weights_dict['bert_encoder_final_input_quantizer_amax']
    embeddings.set_dynamic_range(-dr_out, dr_out)

    # squad_output = network.add_fully_connected(embeddings, 2, Wsquad, Bsquad)
    squad_output = network.add_convolution(embeddings, 2, (1, 1), Wsquad, Bsquad)
    squad_output.name = 'squad_logits'
    logits = squad_output.get_output(0)

    # 1 x 2 x sum_s x 1
    logit_shuffle = network.add_shuffle(logits)
    logit_shuffle.first_transpose = (2, 1, 0, 3)
    logits = logit_shuffle.get_output(0)

    # output shape will be sum_s x 2 (x 1 x 1)
    mark(network, logits, trt.float16)
