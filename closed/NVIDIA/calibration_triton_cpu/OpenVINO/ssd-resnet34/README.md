#  OpenVINO Int8 Quantization for ssd-resnet34
The following instructions are based on Intel's MLPerf v0.7 Inference 
submission: [link](https://github.com/mlcommons/inference_results_v0.7/tree/master/closed/Intel/calibration/OpenVINO/ssd-resnet34)

Please note the modifications that have been made.

#  ssd-resnet34 Quantization

This README file covers quantization process for ssd-resnet34. Please setup the 
prerequisites as instructed at [OpenVINO calibration README](../README.md) before
you attempt the instructions provided here.

# Initialize OpenVINO Environment

To initialize OpenVINO environment, setupvars script must be run. It is located 
at PATH/TO/OPENVINO/bin folder.

#  Instructions on How to Generate IR for ssd-resnet34

Source weight https://zenodo.org/record/3228411/files/resnet34-ssd1200.onnx

For this submission, the NVIDIA MLPerf submission harness is used to run the model.
This harness expects a single output tensor rather than the separate bboxes, labels, and scores output tensors in the ONNX reference model.
As such, a script has been provided to cast the tensors to FP32 and concatenate them.
To run the script:
```
python3 modify_onnx_model.py
```

Run the instruction provided below to generates IR .xml and .bin files. Please 
make sure the path to were OpenVINO has installed matches your installation.
Please also adjust the path to your input model after --input_model option

```
export OPENVINO_INSTALL="PATH/TO/OPENVINO"

python3 ${OPENVINO_INSTALL}/deployment_tools/model_optimizer/mo.py \
	--input_model resnet34-ssd1200_modified.onnx \
	--output_dir FP16 \
	--data_type FP16 \
	--model_name ssd-resnet34_fp16 \
	--input image \
	--mean_values [123.675,116.28,103.53] \
	--scale_values [58.395,57.12,57.375] \
	--input_shape "[1,3,1200,1200]" \
	--keep_shape_ops
```

# Run Post-Training Optimization

Run instruction provided below. Please adjust the path to coco database ([train2017](http://images.cocodataset.org/zips/train2017.zip)) in 
ssd_resnet34_1200_int8_simplified json file.

```
pot -c ssd_resnet34_1200_int8_simplified.json --output-dir results
```

If an error is encountered during pot, make a minor edit to the OpenVINO IR xml file:
change <data_type value="DIR"/> to <data_type value="FP16"/>.

# Notes
+ The calibration list used (annotations/cali.txt) is one required by MLPerf: [coco_cal_images_list](https://github.com/mlperf/inference/blob/master/calibration/COCO/coco_cal_images_list.txt). Please download it and save it as annotations/cali.txt
+ We used ```get_annotations.py``` script to create the annotations for only the calibration list.
