#  OpenVINO Int8 Workflow In a Nutshell

To run OpenVino backend please install first [OpenVino 2021.2] (https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit/choose-download.html)

To run OpenVino inference benchmarks you would need to convert ONNX model into OpenVino IR format. Int8 inference will require additional step, for that you would need to quantize model using calibrations data.


# Converting a Model to Intermediate Representation (IR)

To convert ONNX model to OpenVino IR format use Model Optimizer

```
python <OPENVINO_INSTALL_DIR/deployment_tools/model_optimizer/>mo_onnx.py \
--input_model <path_to_model/>model.onnx \
--model_name 3d_unet_model
```

After that you will get OpenVino IR model represented in 2 files xml and bin format, ready for FP32 inference, copy files to "<path_to_mlperf_3dunet>/build/model" directory.

# Loadgen installation

MLPerf loadgen need to be installed before running model benchmarks or model calibration.

1. git clone https://github.com/mlperf/inference.git --depth 1
2. pip install pybind11
3. cd loadgen; CFLAGS="-std=c++14 -O3" python setup.py develop

# Model Calibration

To run Int8 inference you would need to calibrate model using calibration data and calibration script.

# Post-Training Optimization Toolkit

Before jumping into model calibration install Post-Training Optimization Toolkit, [here] (https://docs.openvinotoolkit.org/latest/pot_README.html) you can find step by step instructions.

# Calibration Data

Prepare calibration data

1. Download images list for [calibration] (https://github.com/mlperf/inference/blob/master/calibration/BraTS/brats_cal_images_list.txt).
2. Run preprocess.py to do data preprocessing for calibration data.

```
python preprocess.py \
--validation_fold_file brats_cal_images_list.txt \
--preprocessed_data_dir build/calibration
```

Calibrate model using ov_calibrate.py

```
python ov_calibrate.py \
--model build/model/3d_unet_model.xml
--model_name 3d_unet_model
--preprocessed_data_dir build/calibration
--int8_directory build/model/calibrated
```