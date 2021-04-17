#  SSD-Resnet34 Quantization

This README file covers quantization process for ssd-resnet34. Please setup the 
prerequisites as instructed at [OpenVINO calibration README](../README.md) before
you attempt the instructions provided here.

# Initialize OpenVINO Environment

To initialize OpenVINO environment, setupvars script must be run. It is located 
at <PATH-TO-OPENVINO>/bin folder.

#  Instructions on How to Generate IR for SSD-Resnet34

Source weight https://zenodo.org/record/3228411/files/resnet34-ssd1200.onnx

Run the instruction provided below to generates IR .xml and .bin files. Please 
make sure the path to were OpenVINO has installed matches your installation.
Please also adjust the path to your input model after --input_model option

```
export OPENVINO_INSTALL="PATH/TO/OPENVINO"

python3 ${OPENVINO_INSTALL}/deployment_tools/model_optimizer/mo.py \
	--input_model <PATH-TO-resnet34-ssd1200.onnx> \
	--data_type FP16 \
	--output_dir <PATH-TO-OUTPUT-DIR> \
	--model_name ssd-resnet34_fp16 \
	--input image \
	--mean_values [123.675,116.28,103.53] \
	--scale_values [58.395,57.12,57.375] \
	--input_shape "[1,3,1200,1200]" \
	--keep_shape_ops
```
Before you proceed to the next step, open the generated .xml file. Search for data_type and
make sure it is set to FP16. If it is set to DIR, change it to FP16 and save the file.

# Run Post-Training Optimization

Run instruction provided below. Please adjust the path to coco database ([train2017](http://images.cocodataset.org/zips/train2017.zip)) in 
ssd_resnet34_1200_int8_simplified json file.

```
pot -c ssd_resnet34_1200_int8_simplified.json --output-dir results
```

# Notes
+ The calibration list used (annotations/cali.txt) is one required by MLPerf: [coco_cal_images_list](https://github.com/mlperf/inference/blob/master/calibration/COCO/coco_cal_images_list.txt). Please download it and save it as annotations/cali.txt
+ We used ```get_annotations.py``` script to create the annotations for only the calibration list.
