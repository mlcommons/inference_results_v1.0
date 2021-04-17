import types
from typing import Tuple, List, Union, Dict
import copy

import numpy as np
import torch

import libpymo
import aimet_torch
import aimet_torch.onnx_utils
from aimet_torch import utils
import aimet_torch.bias_correction
from aimet_torch.qc_quantize_op import QcQuantizeWrapper, QcPostTrainingWrapper
from aimet_common.defs import QuantScheme

import models.ssd_r34 as ssd_r34
from modifications.ssd_r34_mod import Concat as ConcatMod
aimet_torch.onnx_utils.map_torch_types_to_onnx[ConcatMod] = 'Concat'

def my_compute_encoding_for_given_bitwidth(data: np.ndarray, bitwidth: int, quant_scheme: QuantScheme,
                                        is_symmetric: bool) -> Dict:
    """
    Return encoding dictionary for given bitwidth
    :param data: Numpy data
    :param bitwidth: bitwidth (4-31) to use for quantizing data
    :param quant_scheme: Quantization scheme
    :param is_symmetric: True if symmetric encodings is used, False otherwise
    :return: Encoding Dictionary
    """
    # Create Encodings Analyzer and collect statistical data to compute encodings
    # Since the data is numpy array and on CPU memory, useCuda is False
    encoding_analyzer = libpymo.EncodingAnalyzerForPython(quant_scheme)
    encoding_analyzer.updateStats(data, False)

    encoding, is_encoding_valid = encoding_analyzer.computeEncoding(bitwidth, is_symmetric, False, False)

    if is_encoding_valid:
        return {'min': encoding.min,
                'max': encoding.max,
                'scale': encoding.delta,
                'offset': int(encoding.offset),
                'bitwidth': encoding.bw,
                'is_symmetric': str(is_symmetric)}

    return {}

def my_map_onnx_nodes_to_pytorch(torch_model: torch.nn.Module, dummy_input,
                                onnx_ordered_list):
    torch_ordered_list = aimet_torch.utils.get_ordered_list_of_modules(torch_model, dummy_input)

    torch_index = 0
    onnx_index = 0

    num_onnx_nodes_to_map_to_same_torch_node = 0
    while torch_index < len(torch_ordered_list):
        # If few PyTorch ops are not mapped to ONNX ops
        if onnx_index >= len(onnx_ordered_list):
            aimet_torch.onnx_utils._logger.warning('All ONNX ops were exhausted but few PyTorch ops did not get mapped to a '
                            'corresponding ONNX op')
            break
        name, module = torch_ordered_list[torch_index]

        if isinstance(module, tuple(aimet_torch.onnx_utils.torch_types_to_ignore)):
            torch_index += 1
            continue
        if onnx_ordered_list[onnx_index].op_type in 'Concat' and len(onnx_ordered_list[onnx_index].input) < 6:
            pass 
        elif onnx_ordered_list[onnx_index].op_type in aimet_torch.onnx_utils.map_torch_types_to_onnx[type(module)]:
            aimet_torch.onnx_utils._logger.debug('Found a match: %r -> %r', onnx_ordered_list[onnx_index].op_type, name)
            onnx_ordered_list[onnx_index].name = name

            if num_onnx_nodes_to_map_to_same_torch_node == 0:
                num_onnx_nodes_to_map_to_same_torch_node = aimet_torch.onnx_utils.OnnxSaver.get_num_onnx_nodes_to_map(module)

            num_onnx_nodes_to_map_to_same_torch_node = num_onnx_nodes_to_map_to_same_torch_node - 1
            if num_onnx_nodes_to_map_to_same_torch_node == 0:
                torch_index += 1

        onnx_index += 1

def my_update_param_encodings_dict_for_layer(self, layer: torch.nn.Module, layer_name: str,
											param_encodings: Dict, valid_param_set: set):
	"""
	:param layer: layer as torch.nn.Module
	:param layer_name : Name of the layer
	:param param_encodings: dictionary of param encodings
	:param valid_param_set: a set of valid param input names in model
	"""

	disabled_param_quantizers = []
	for orig_param_name, param_quantizer in layer.param_quantizers.items():
		param_name = layer_name + '.' + orig_param_name
		if param_quantizer.enabled:
			if param_name in valid_param_set:
				encoding = utils.create_encoding_dict(param_quantizer.encoding,
														param_quantizer.use_symmetric_encodings)
				param_encodings[param_name] = [encoding]
			else:
				logger.error('Param tensor {%s} not found in valid param set', param_name)
		else:
			disabled_param_quantizers.append(orig_param_name)

	# retrieve the appropriate param generator
	if isinstance(layer, QcQuantizeWrapper):
		# pylint: disable=protected-access
		named_parameters = layer._module_to_wrap.named_parameters()
	else:
		named_parameters = layer.named_parameters(recurse=False)

	for name, param in named_parameters:
		# if the param quantizer was disabled generate encoding assuming bitwidth of 32
		if name in disabled_param_quantizers:
			param_name = layer_name + '.' + name
			param_quantizer = layer.param_quantizers[name]
			param_data = param.cpu().detach().numpy()
			encoding = utils.compute_encoding_for_given_bitwidth(param_data, 31, param_quantizer.quant_scheme,
																	param_quantizer.use_symmetric_encodings)
			param_encodings[param_name] = [encoding]

def my_correct_bias(model: torch.nn.Module, quant_params,
                 num_quant_samples: int, data_loader, num_bias_correct_samples: int,
                 conv_bn_dict = None,
                 perform_only_empirical_bias_corr: bool = True,
                 layers_to_ignore = None,
                 quantizer_modifications=None):

    if layers_to_ignore is None:
        layers_to_ignore = []

    # Find batch size and shape of input tensor
    batch_size, input_shape = aimet_torch.utils.get_input_shape_batch_size(data_loader)

    # Rounding up number of samples to batch size
    n_batches_bias_correction = int(np.ceil(num_bias_correct_samples / batch_size))
    n_batches_quantization = int(np.ceil(num_quant_samples / batch_size))

    data_loader_n_samples_bias_corr = aimet_torch.utils.IterFirstX(data_loader, n_batches_bias_correction)
    data_loader_n_samples_quant = aimet_torch.utils.IterFirstX(data_loader, n_batches_quantization)

    # TODO: Remove wrapper function
    # Create a wrapping function for data loader for quantization
    def pass_data_through_model(model, early_stopping_iterations=None, use_cuda=False):
        # pylint: disable=unused-argument
        # forward pass for given number of batches for model
        for (images_in_one_batch, _) in data_loader_n_samples_quant:
            aimet_torch.bias_correction.forward_pass(model, images_in_one_batch)

    ordered_conv_linear_nodes = aimet_torch.utils.get_ordered_lists_of_conv_fc(model, input_shape)

    if conv_bn_dict is None:
        conv_bn_dict = aimet_torch.bias_correction.find_all_conv_bn_with_activation(model, input_shape)

    # Create a copy of the model as reference model
    model_copy = copy.deepcopy(model)

    # Add bias for all the layers whose bias is None
    for name, module in ordered_conv_linear_nodes:
        if module.bias is None:
            if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
                output_size = module.out_channels
            elif isinstance(module, torch.nn.Linear):
                output_size = module.out_features
            module.bias = torch.nn.Parameter(torch.zeros(output_size))
            module.bias.data = module.bias.data.to(device=module.weight.device)

    # Quantize full model
    dummy_tensors = aimet_torch.utils.create_rand_tensors_given_shapes(input_shape)
    dummy_tensors = [tensor.to(aimet_torch.utils.get_device(model)) for tensor in dummy_tensors]
    q = aimet_torch.quantsim.QuantizationSimModel(model=model, quant_scheme=quant_params.quant_scheme,
                                  rounding_mode=quant_params.round_mode,
                                  default_output_bw=quant_params.act_bw,
                                  default_param_bw=quant_params.weight_bw,
                                  in_place=True,
                                  dummy_input=dummy_tensors, config_file=quant_params.config_file)

    # make sure  model got updated in-place before we use it for bc updates
    assert(q.model is model)

    if quantizer_modifications is not None:
        quantizer_modifications(q)

    # updates to skip_output_activation and layers_to_ignore
    for name, module in model.named_modules():
        # Skip all layer's output quantization
        if isinstance(module, QcQuantizeWrapper):
            module.output_quantizers[0].enabled = False

    q.compute_encodings(pass_data_through_model, None)

    # For first conv layer, perform analytical bc if perform_only_empirical_bias_corr is set to False
    # and layer is not marked to be ignored during bc.
    if not perform_only_empirical_bias_corr:
        module_name, module = ordered_conv_linear_nodes[0]
        if module not in layers_to_ignore:
            aimet_torch.bias_correction.logger.info('Correcting layer %s using Analytical Bias Correction', module_name)
            quantize_layer = aimet_torch.utils.get_layer_by_name(model, module_name)
            aimet_torch.bias_correction.call_analytical_mo_correct_bias(quantize_layer, None, None)
            aimet_torch.bias_correction.logger.info('Corrected bias for the layer')
            ordered_conv_linear_nodes.pop(0)

    for module_name, module in ordered_conv_linear_nodes:
        # Ignore all layers which are skipped by user
        if module in layers_to_ignore or module_name in layers_to_ignore:
            continue
        else:
            # make sure module is in the model used by qsim.
            assert(module in list(q.model.modules()))
            # Analytical Bias Correction is only done for Conv layers
            reference_layer = aimet_torch.utils.get_layer_by_name(model_copy, module_name)
            quantize_layer = aimet_torch.utils.get_layer_by_name(model, module_name)

            if module in conv_bn_dict.keys():

                bn_layer_info = conv_bn_dict[module]

                if perform_only_empirical_bias_corr or bn_layer_info is None or bn_layer_info.input_bn is None:
                    aimet_torch.bias_correction.logger.info('Correcting layer %s using Empirical Bias Correction', module_name)
                    bias_correction = libpymo.BiasCorrection()

                    # Get output from quantized model and reference model

                    for images_in_one_batch, _ in data_loader_n_samples_bias_corr:
                        reference_output_batch = aimet_torch.bias_correction.get_output_data(reference_layer, model_copy, images_in_one_batch)
                        quantized_model_output_batch = aimet_torch.bias_correction.get_output_data(quantize_layer, model, images_in_one_batch)

                        if isinstance(reference_layer, torch.nn.Linear):
                            extended_shape = np.concatenate((reference_output_batch.shape, np.array([1, 1])))
                            reference_output_batch = reference_output_batch.reshape(extended_shape)
                            quantized_model_output_batch = quantized_model_output_batch.reshape(extended_shape)

                        bias_correction.storePreActivationOutput(reference_output_batch)
                        bias_correction.storeQuantizedPreActivationOutput(quantized_model_output_batch)

                    aimet_torch.bias_correction.call_empirical_mo_correct_bias(module, bias_correction)

                else:
                    aimet_torch.bias_correction.logger.info('Correcting layer %s using Analytical Bias Correction', module_name)
                    aimet_torch.bias_correction.call_analytical_mo_correct_bias(quantize_layer, bn_layer_info.input_bn,
                                                    bn_layer_info.in_activation_type)

                aimet_torch.bias_correction.logger.info('Corrected bias for the layer')

    aimet_torch.save_utils.SaveUtils.remove_quantization_wrappers(model)

    aimet_torch.bias_correction.logger.info('Completed bias correction')

def quantizer_modifications(quantizer):
    from modifications import ssd_r34_mod
    quantizer._update_param_encodings_dict_for_layer = types.MethodType(my_update_param_encodings_dict_for_layer, quantizer)

    asymmetric_weight_quantization = [
        'model.layer2.0.4.conv2',
        'model.layer2.0.5.conv1',
        'model.layer2.0.5.conv2'
    ]
    asymmetric_actvn_quantization = [
        'catModule1',
        'catModule2',
        'conf.0',
        'conf.1',
        'conf.2',
        'conf.3',
        'model.layer2.0.4.conv2',
        'model.layer2.0.4.relu2',
        'model.layer2.0.5.relu',
        'model.layer2.0.5.conv2'
    ]

    for name,module in quantizer.model.named_modules():
        if name in asymmetric_weight_quantization:
            module.param_quantizers['weight'].use_symmetric_encodings = False
        if name in asymmetric_actvn_quantization:
            module.output_quantizer.use_symmetric_encodings = False

utils.compute_encoding_for_given_bitwidth = my_compute_encoding_for_given_bitwidth
aimet_torch.onnx_utils.OnnxSaver.map_onnx_nodes_to_pytorch = my_map_onnx_nodes_to_pytorch
aimet_torch.bias_correction.correct_bias = my_correct_bias
