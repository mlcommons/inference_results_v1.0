#  Resnet50 Quantization

This README file covers quantization process for resnet50. Please setup the 
prerequisites as instructed at [OpenVINO calibration README](../README.md) before
you attempt the instructions provided here.

# Initialize OpenVINO Environment

To initialize OpenVINO environment, setupvars script must be run. It is located 
at <PATH-TO-OPENVINO>/bin folder.

# Generate IR for Resnet50

The starting FP32 Weight is at "https://zenodo.org/record/2535873/files/resnet50_v1.pb"

Run the instruction provided below tp generates IR .xml and .bin files. Please 
make sure the path to were OpenVINO has installed matches your installation.
Please also adjust the path to your input model using --input_model option.
Keep the order in which command line options appear in the command below.

```
export OPENVINO_INSTALL="PATH/TO/OPENVINO"

python3 ${OPENVINO_INSTALL}/deployment_tools/model_optimizer/mo_tf.py \
  	--input_model <PATH-TO-resnet50_v1.pb> \
	--data_type FP16 \
	--output_dir <PATH-TO-OUTPUT-DIR> \
	--input_shape [1,224,224,3] \
	--mean_values "[123.68, 116.78, 103.94]" \
	--model_name resnet50_fp16 \
	--output softmax_tensor
```
Before you proceed to the next step, open the generated .xml file. Search for data_type and
make sure it is set to FP16. If it is set to DIR, change it to FP16 and save the file.

# Run Post-Training Optimization

Run instruction provided below. Please adjust the path to imagenet database within
resnet50.yml file.

```
pot -c resnet50.json --output-dir results
```

