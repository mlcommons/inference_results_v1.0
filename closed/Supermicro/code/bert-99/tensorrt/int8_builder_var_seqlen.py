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

def add_small_tile_gemm_fc(network, input_tensor, input_channels, output_channels,
    layer_name, weight, bias, input_dr, output_dr, use_gelu=False, fairshare_cache_size=-1):
    """ Build one plugin layer of the Small-Tile GEMM kernel"""
    print("Replacing {} with small-tile GEMM plugin, with fairshare cache size {}.".\
        format(layer_name, fairshare_cache_size))
    plugin_name = "SmallTileGEMM_TRT"
    plugin_layer_name = layer_name + plugin_name
    plugin_version = '1'
    plugin_creator = trt.get_plugin_registry().\
        get_plugin_creator(plugin_name, plugin_version, '')
    if plugin_creator is None:
        raise Exception("Cannot find small tile GEMM plugin creator for top_mlp")

    scale = np.ones([output_channels], dtype=np.float32)

    fields = []
    fields.append(trt.PluginField("inputChannels", np.array([input_channels],
        dtype=np.int32), trt.PluginFieldType.INT32))
    fields.append(trt.PluginField("weight", weight, trt.PluginFieldType.FLOAT32))
    fields.append(trt.PluginField("bias", bias, trt.PluginFieldType.FLOAT32))
    fields.append(trt.PluginField("scale", scale, trt.PluginFieldType.FLOAT32))
    fields.append(trt.PluginField("fairShareCacheSize", np.array([fairshare_cache_size],
        dtype=np.int32), trt.PluginFieldType.INT32))
    fields.append(trt.PluginField("dynamicRanges", np.array([input_dr, output_dr],
        dtype=np.float32), trt.PluginFieldType.FLOAT32))

    if use_gelu:
        rescale = np.ones([output_channels], dtype=np.float32)
        fields.append(trt.PluginField("rescale", rescale, trt.PluginFieldType.FLOAT32))
        fields.append(trt.PluginField("epilogueScaleBiasGelu", np.array([1],
            dtype=np.int32), trt.PluginFieldType.INT32))
    else:
        fields.append(trt.PluginField("epilogueScaleBias", np.array([1],
            dtype=np.int32), trt.PluginFieldType.INT32))
        
    fields = trt.PluginFieldCollection(fields)

    plugin = plugin_creator.create_plugin(plugin_layer_name, fields)
    if plugin is None:
        raise Exception("Cannot create BERT Small-Tile GEMM plugin for {}.".format(plugin_layer_name))

    plugin_layer = network.add_plugin_v2([input_tensor], plugin)

    return plugin_layer


def bert_encoder_layer_int8_var_seqlen(cfg, max_seqlen, weights_dict, network,
    input_tensor, cu_seqlens, layer, mask, use_small_tile_gemm_plugin,
    gemm_plugin_fairshare_cache_size):
    """Builds one encoder layer in INT8 with var seqlen.
    Sets the dynamic ranges extracted from the qat checkpoint."""

    plg_registry = trt.get_plugin_registry()
    qkv_plg_creator = plg_registry.get_plugin_creator("CustomQKVToContextPluginDynamic", "2", "")
    pc_skln = plg_registry.get_plugin_creator("CustomSkipLayerNormPluginDynamic", "2", "")
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

    Wqkv = np.ascontiguousarray(Wqkv.reshape((3, N, H, N, H)).transpose((1, 0, 2, 3, 4)))
    Bqkv = np.ascontiguousarray(Bqkv.reshape((3, N, H)).transpose((1, 0, 2)))

    if use_small_tile_gemm_plugin:
        # Replace QKV FC with GEMM plugin
        # [BS, 1024, 1, 1] -> [BS, 3072, 1, 1]
        fc_qkv_input_channels = input_tensor.shape[1]
        fc_qkv_layer_name = prefix + 'fc_qkv'
        fc_qkv_plugin = add_small_tile_gemm_fc(network, input_tensor, fc_qkv_input_channels,
            cfg.qkv_size, fc_qkv_layer_name, Wqkv, Bqkv, dr_input, dr_qkv,
            use_gelu=False, fairshare_cache_size=gemm_plugin_fairshare_cache_size)
        fc_qkv_out = fc_qkv_plugin.get_output(0)
    else:
        fc_qkv = network.add_convolution(input_tensor, cfg.qkv_size, (1, 1), Wqkv, Bqkv)
        fc_qkv.name = prefix + 'fc_qkv'
        fc_qkv_out = fc_qkv.get_output(0)

    fc_qkv_out.name = prefix + 'attention_self_qkv_mult'
    fc_qkv_out.set_dynamic_range(-dr_qkv, dr_qkv)
    # QKV2CTX
    dr_probs = weights_dict[prefix + 'attention_self_av_a_input_quantizer_amax']
    dq_probs = dr_probs / 127.0
    pf_type = trt.PluginField("type_id", np.array([int(trt.int8)], np.int32), trt.PluginFieldType.INT32)
    pf_hidden_size = trt.PluginField("hidden_size", np.array([cfg.hidden_size], np.int32), trt.PluginFieldType.INT32)
    pf_num_heads = trt.PluginField("num_heads", np.array([cfg.N], np.int32), trt.PluginFieldType.INT32)
    pf_has_mask = trt.PluginField("has_mask", np.array([1], np.int32), trt.PluginFieldType.INT32)
    pf_dq_probs = trt.PluginField("dq_probs", np.array([dq_probs], np.float32), trt.PluginFieldType.FLOAT32)
    pf_var_seqlen = trt.PluginField("var_seqlen", np.array([int(1)], np.int32), trt.PluginFieldType.FLOAT32)

    pfc = trt.PluginFieldCollection([pf_hidden_size, pf_num_heads, pf_has_mask, pf_type, pf_dq_probs, pf_var_seqlen])
    qkv2ctx_plug = qkv_plg_creator.create_plugin("qkv2ctx", pfc)

    dr_ctx = weights_dict[prefix + 'attention_output_dense_input_amax']
    qkv2ctx_layer = network.add_plugin_v2([fc_qkv_out, mask, cu_seqlens, max_seqlen], qkv2ctx_plug)
    qkv2ctx_layer.name = prefix + 'qkv_to_ctx'
    qkv2ctx_out = qkv2ctx_layer.get_output(0)
    qkv2ctx_out.set_dynamic_range(-dr_ctx, dr_ctx)
    # FC AOUT
    dr_fc_aout = weights_dict[prefix + 'attention_output_add_local_input_quantizer_amax']
    Waout = weights_dict[prefix + 'attention_output_dense_kernel']
    Baout = weights_dict[prefix + 'attention_output_dense_bias']
    
    if use_small_tile_gemm_plugin:
        # Replace fc aout with small-Tile GEMM
        # [BS, 1024, 1, 1] -> [BS, 1024, 1, 1]
        fc_aout_input_channels = qkv2ctx_out.shape[1]
        fc_aout_layer_name = prefix + 'fc_aout'
        fc_aout_plugin = add_small_tile_gemm_fc(network, qkv2ctx_out, fc_aout_input_channels,
            cfg.hidden_size, fc_aout_layer_name, Waout, Baout, dr_ctx, dr_fc_aout,
            use_gelu=False, fairshare_cache_size=gemm_plugin_fairshare_cache_size)
        fc_aout_out = fc_aout_plugin.get_output(0)
    else:
        fc_aout = network.add_convolution(qkv2ctx_out, cfg.hidden_size, (1, 1), Waout, Baout)
        fc_aout.precision = dtype
        fc_aout.name = prefix + 'fc_aout'
        fc_aout_out = fc_aout.get_output(0)
    
    fc_aout_out.dtype = dtype
    fc_aout_out.set_dynamic_range(-dr_fc_aout, dr_fc_aout)
    # Skip-Layernorm 1
    dr_skln1 = weights_dict[prefix + 'intermediate_dense_input_amax']
    pf_ld = trt.PluginField("ld", np.array([cfg.hidden_size], np.int32), trt.PluginFieldType.INT32)
    pf_type = trt.PluginField("type_id", np.array([int(dtype)], np.int32), trt.PluginFieldType.INT32)
    pf_beta = trt.PluginField("beta", weights_dict[prefix + 'attention_output_layernorm_beta'], trt.PluginFieldType.FLOAT32)
    pf_gamma = trt.PluginField("gamma", weights_dict[prefix + 'attention_output_layernorm_gamma'], trt.PluginFieldType.FLOAT32)
    pf_bias = trt.PluginField("bias", Baout, trt.PluginFieldType.FLOAT32)
    fields = [pf_ld, pf_beta, pf_gamma, pf_type]
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

    dr_gelu = weights_dict[prefix + 'output_dense_input_amax']

    if use_small_tile_gemm_plugin:
        # Replace FC MID with small-tile GEMM kernel (with Gelu epilogue)
        # [BS, 1024, 1, 1] -> [BS, 4096, 1, 1]
        fc_mid_input_channels = skln1_out.shape[1]
        fc_mid_layer_name = prefix + 'fc_mid_gelu'
        fc_mid_plugin = add_small_tile_gemm_fc(network, skln1_out, fc_mid_input_channels,
            cfg.mid_size, fc_mid_layer_name, Wmid, Bmid, dr_skln1, dr_gelu,
            use_gelu=True, fairshare_cache_size=gemm_plugin_fairshare_cache_size)
        gelu_out = fc_mid_plugin.get_output(0)
    else:
        fc_mid = network.add_convolution(skln1_out, cfg.mid_size, (1, 1), Wmid, Bmid)
        fc_mid.name = prefix + 'fc_mid'
        fc_mid_out = fc_mid.get_output(0)
        # GELU
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
    pf_bias = trt.PluginField("bias", Bout, trt.PluginFieldType.FLOAT32)
    fields = [pf_ld, pf_beta, pf_gamma, pf_type]
    pfc = trt.PluginFieldCollection(fields)
    skipln_plug = pc_skln.create_plugin("skipln", pfc)

    skln1_out.dtype = dtype  # It does not build without setting this here, in addition to above. WHY??!?!

    skipln_inputs = [fc_out_out, skln1_out]
    skln2 = network.add_plugin_v2(skipln_inputs, skipln_plug)
    skln2.name = prefix + 'skln_2'
    skln2_out = skln2.get_output(0)

    return skln2_out


def bert_squad_int8_var_seqlen(network, weights_dict, cfg, input_shape,
    cu_seqlens_shape, use_small_tile_gemm_plugin, gemm_plugin_fairshare_cache_size):
    """Create BERT network with INT8, var seqlen."""

    # instantiate all the plugins
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

    # dummy input used to indicate maximum sequence length to plugins
    max_seqlen = network.add_input(name="max_seqlen", dtype=trt.int32, shape=(-1,))

    inputs = [input_ids, segment_ids, cu_seqlens, max_seqlen]
    emb_layer = network.add_plugin_v2(inputs, embln_plugin)
    emb_layer.name = 'embln'

    embeddings = emb_layer.get_output(0)

    mask = emb_layer.get_output(1)
    embeddings.dtype = dtype
    mask.set_dynamic_range(-1, 1)

    layer = 0
    for layer in range(cfg.L):
        embeddings = bert_encoder_layer_int8_var_seqlen(cfg, max_seqlen, weights_dict,
            network, embeddings, cu_seqlens, layer, mask, use_small_tile_gemm_plugin,
            gemm_plugin_fairshare_cache_size)

    Wsquad = weights_dict['cls_squad_output_weights']
    Bsquad = weights_dict['cls_squad_output_bias']

    dr_out = weights_dict['bert_encoder_final_input_quantizer_amax']
    embeddings.set_dynamic_range(-dr_out, dr_out)

    #squad_output = network.add_fully_connected(embeddings, 2, Wsquad, Bsquad)
    squad_output = network.add_convolution(embeddings, 2, (1, 1), Wsquad, Bsquad)
    squad_output.name = 'squad_logits'
    logits = squad_output.get_output(0)

    # output shape will be sum_s x 2 (x 1 x 1)
    mark(network, logits, trt.float16)
