#  OpenVINO Int8 Quantization for resnet50
The following instructions are copied from Intel's MLPerf v0.7 Inference 
submission: [link](https://github.com/mlcommons/inference_results_v0.7/tree/master/closed/Intel/calibration/OpenVINO/resnet50)

#  resnet50 Quantization

This README file covers quantization process for resnet50. Please setup the 
prerequisites as instructed at [OpenVINO calibration README](../README.md) before
you attempt the instructions provided here.

# Initialize OpenVINO Environment

To initialize OpenVINO environment, setupvars script must be run. It is located 
at PATH/TO/OPENVINO/bin folder.

# Generate IR for resnet50

The starting FP32 Weight is at "https://zenodo.org/record/2535873/files/resnet50_v1.pb"

Run the instruction provided below to generate OpenVINO IR .xml and .bin files. Please 
make sure the path to where OpenVINO has been installed matches your installation.
Please also adjust the path to your input model using --input_model option.

```
export OPENVINO_INSTALL="PATH/TO/OPENVINO"

python3 ${OPENVINO_INSTALL}/deployment_tools/model_optimizer/mo_tf.py \
  --input_model PATH/TO/resnet50_v1.pb \
	--input_shape [1,224,224,3] \
	--output_dir FP16 \
	--mean_values "[123.68, 116.78, 103.94]" \
	--data_type FP16 \
	--model_name resnet50_fp16 \
	--output softmax_tensor
```

# Run Post-Training Optimization

Run instruction provided below. Please adjust the path to imagenet database within
resnet50.yml file.

```
pot -c resnet50.json --output-dir results
```

