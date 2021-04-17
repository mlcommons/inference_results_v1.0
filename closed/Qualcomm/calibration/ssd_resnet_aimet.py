import sys
import os
import types
import yaml
import json
import pickle
from tqdm import tqdm
from datetime import datetime

import torch
import aimet_torch
import aimet_torch.batch_norm_fold
import aimet_torch.cross_layer_equalization
import aimet_torch.bias_correction
from aimet_torch.quantsim import QuantizationSimModel, QuantParams

import inference
from inference.vision.classification_and_detection.python.models import ssd_r34
from inference.vision.classification_and_detection.python.dataset import pre_process_coco_resnet34

import modifications
from modifications.ssd_r34_mod import model_modifications
from modifications.aimet_mod import quantizer_modifications
from modifications.inference_mod import CocoMod

usage_str = \
'Usage: \n \
      python ssd_resnet_aimet.py <path/to/resnet34-ssd1200.pytorch> <path/to/calibration-annotations> <path/to/calibration-images> \n \
   or python ssd_resnet_aimet.py show-params <path/to/params-file.pkl>'

class Params:
	def __init__(self, 
		model_file, encodings_dataset_list=None, encodings_dataset_path=None, 
	):
		self.show_params = False
		self.model_score_threshold = 0.05
		self.model_nms_threshold = 0.5
		self.model_max_output = 200
		self.post_proc_score_thresh = 0.05
		self.input_shape = [1,3,1200,1200]
		self.input_shape_tuple = tuple(self.input_shape)
		self.image_size = [1200,1200,3]
		self.dataset_name = 'coco-1200-pt'
		self.image_format = 'NCHW'
		self.use_label_map = True
		self.pre_proc_func = pre_process_coco_resnet34
		self.do_bias_correction = False
		self.cache = 0
		self.count_enc = 500
		self.count_full = 5000
		self.count_bc = 25
		self.log_path = 'output'
		self.filename_prefix = 'ssd_resnet34_aimet'
		self.my_filename = self.filename_prefix + '_params.pkl'
		self.quant_scheme = 'tf'
		self.config_file = 'configs/aic100_config_symmetric.json'
		self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu') 
		self.default_bitwidth = 8
		self.rounding_mode = 'nearest'
		self.timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
		self.log_path = os.path.join(self.log_path, self.timestamp)

		self.model_file = model_file
		self.encodings_dataset_list = encodings_dataset_list
		self.encodings_dataset_path = encodings_dataset_path

	def print_params(self):
		print({x: getattr(self, x) for x in dir(self) if not x.startswith('__')})

class CocoTorchDataset(torch.utils.data.Dataset):
	def __init__(self, device, *args, **kwargs):
		super(CocoTorchDataset, self).__init__()
		self.coco_ds = CocoMod(*args, **kwargs)
		self.device = device

	def get_item(self, nr):
		return self.coco_ds.get_item(nr)

	def get_item_loc(self, nr):
		return self.coco_ds.get_item_loc(nr)

	def preprocess(self, use_cache=True):
		return self.coco_ds.preprocess(use_cache)

	def get_item_count(self):
		return self.coco_ds.get_item_count()

	def get_list(self):
		return self.coco_ds.get_list()

	def load_query_samples(self, sample_list):
		return self.coco_ds.load_query_samples(sample_list)

	def unload_query_samples(self, sample_list):
		return self.coco_ds.unload_query_samples(sample_list)

	def get_samples(self, id_list):
		data, labels = self.coco_ds.get_samples(id_list)
		data = torch.cat(data, dim=0).to(self.device)
		return data, labels

	def get_item_loc(self, id):
		return self.coco_ds.get_item_loc(id)

	def __len__(self):
		return self.get_item_count()

	def __getitem__(self, idx):
		return self.get_item(idx)

	@staticmethod
	def collateFn(batch):
		return batch[0]

	@property
	def image_ids(self):
		return self.coco_ds.image_ids

	@property
	def image_sizes(self):
		return self.coco_ds.image_sizes

	@property
	def annotation_file(self):
		return self.coco_ds.annotation_file

def parse_args():
	print('Calling with args: ', sys.argv[1:])
	if sys.argv[1] in ['help', '--help', '-h']:
		raise IndexError
	elif sys.argv[1] == 'show-params':
		if len(sys.argv) < 3:
			raise IndexError("Did not provide path to parameters file.")
		with open(sys.argv[2], 'rb') as f:
			params = pickle.load(f)
			params.show_params = True
		return params
	if len(sys.argv) < 2:
		raise IndexError("Did not provide path to ssd-resnet34 model.")
	if len(sys.argv) < 3:
		raise IndexError("Did not provide path to calibration set annotations.")
	elif len(sys.argv) < 4:
		raise IndexError("Did not provide path to calibration images.")

	model_file, encodings_dataset_list, encodings_dataset_path = sys.argv[1], sys.argv[2], sys.argv[3]
	params = Params(
		model_file, encodings_dataset_list, encodings_dataset_path, 
	)

	return params

def create_encoder_dataset(params, return_type='dataset'):
	enc_ds = CocoTorchDataset(
		params.device,
		data_path=params.encodings_dataset_path,
		image_list=params.encodings_dataset_list,
		name=params.dataset_name,
		image_format=params.image_format,
		pre_process=params.pre_proc_func,
		use_cache=params.cache,
		count=params.count_enc, 
		image_size=params.image_size,
		use_label_map=params.use_label_map
	)
	if return_type == 'dataset':
		return enc_ds
	elif return_type == 'dataloader':
		enc_dl = torch.utils.data.DataLoader(
			enc_ds, 
			batch_size=1, 
			shuffle=False, 
			num_workers=0, 
			sampler=None,
			pin_memory=False,
			collate_fn=CocoTorchDataset.collateFn
		)
		return enc_dl

def optimize_model(model, params):
	model.cpu()
	folded_pairs = aimet_torch.batch_norm_fold.fold_all_batch_norms(model, params.input_shape_tuple)
	bn_dict = {}
	for conv_bn in folded_pairs:
		bn_dict[conv_bn[0]] = conv_bn[1]
	cls_set_info_list = aimet_torch.cross_layer_equalization.CrossLayerScaling.scale_model(model, params.input_shape_tuple)
	aimet_torch.cross_layer_equalization.HighBiasFold.bias_fold(cls_set_info_list, bn_dict)
	model.to(params.device)

	if params.do_bias_correction:
		enc_dl = create_encoder_dataset(params, return_type='dataloader')

		quant_params = QuantParams(
			weight_bw=params.default_bitwidth, act_bw=params.default_bitwidth, 
			round_mode=params.rounding_mode, quant_scheme=params.quant_scheme, 
			config_file=params.config_file
		)
		aimet_torch.bias_correction.correct_bias(
			model, 
			quant_params, 
			num_quant_samples=len(enc_dl), 
			data_loader = enc_dl, 
			num_bias_correct_samples=params.count_bc,
			quantizer_modifications=quantizer_modifications
		)

def export_and_generate_encodings(model, params):
	os.makedirs(params.log_path)

	enc_ds = create_encoder_dataset(params, return_type='dataset')
			
	def evaluator_enc(model, iterations):
		for query_id in tqdm(range(enc_ds.get_item_count())):
			query_ids = [query_id]
			enc_ds.load_query_samples(query_ids)
			img, label = enc_ds.get_samples(query_ids)
			with torch.no_grad():
				_ = model(img)
			enc_ds.unload_query_samples(query_ids)

	quantizer = QuantizationSimModel(
		model=model,
		input_shapes=params.input_shape_tuple,
		quant_scheme=params.quant_scheme, 
		rounding_mode=params.rounding_mode, 
		default_output_bw=params.default_bitwidth, 
		default_param_bw=params.default_bitwidth, 
		in_place=False, 
		config_file=params.config_file
	)
	quantizer_modifications(quantizer)
	quantizer.compute_encodings(
		forward_pass_callback=evaluator_enc, 
		forward_pass_callback_args=1
	)

	quantizer.export(
		path = params.log_path, 
		filename_prefix = params.filename_prefix, 
		input_shape = params.input_shape_tuple
	)
	
	input_file = os.path.join(params.log_path, '%s.encodings' % str(params.filename_prefix))

	remap_bitwidth_to_32(input_file)

	with open(os.path.join(params.log_path, params.my_filename), 'wb') as f:
		pickle.dump(params, f)

	return quantizer

def remap_bitwidth_to_32(encodings_file):
	with open(encodings_file) as f:
		encodings_dict = json.load(f)
	with open(encodings_file + '.yaml') as f:
		encodings_dict_yaml = yaml.load(f, Loader=yaml.FullLoader)

	for module_name, module_encodings in encodings_dict['param_encodings'].items():
		if module_name.endswith('bias'):
			module_encodings[0]['bitwidth'] = 32
	for module_name, module_encodings in encodings_dict_yaml['param_encodings'].items():
		if module_name.endswith('bias'):
			module_encodings[0]['bitwidth'] = 32

	with open(encodings_file, 'w') as f:
		json.dump(encodings_dict, f, sort_keys=True, indent=4)
	with open(encodings_file + '.yaml', 'w') as f:
		yaml.dump(encodings_dict, f, default_flow_style=False, allow_unicode=True)

def main():
	# parameters
	try:
		params = parse_args()
	except IndexError as e:
		print(str(e) + '\n' + usage_str)
		return

	# different script pathway to inspect parameters post-output; used for sanity-checking 
	if params.show_params:
		params.print_params()
		return

	# load model
	model = torch.load(params.model_file, map_location=params.device)
	model.eval()

	# model modifications
	model_modifications(model, params, mod_type='export')

	# apply aimet
	optimize_model(model, params)

	# generate encodings
	quantizer = export_and_generate_encodings(model, params)

if __name__ == "__main__":
	main()
