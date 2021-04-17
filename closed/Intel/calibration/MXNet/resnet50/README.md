# MXNet ResNet50-v1.5 Calibration for MLPerf v0.7 inference

## 1. Description

This file describes the quantization flow of MXNet for the Resnet50-v1.5 model, Intel® Low Precision Optimization Tool is used for the quantizaiton, refer [here](https://github.com/intel/lp-opt-tool) for more detail about the tool.

The calibration dataset was based on mlperf provided list in [here](https://github.com/mlperf/inference/blob/master/calibration/ImageNet/cal_image_list_option_1.txt)


## 2. Prerequisites before Quantization
### 2.1 Install MKL, MXNet, ONNX
```bash
# Install MKL
sudo bash
# <type your user password when prompted.  this will put you in a root shell>
# cd to /tmp where this shell has write permission
cd /tmp
# now get the key:
wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
# now install that key
apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
# now remove the public key file exit the root shell
rm GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
exit
sudo sh -c 'echo deb https://apt.repos.intel.com/mkl all main > /etc/apt/sources.list.d/intel-mkl.list'
sudo apt update
sudo apt install intel-mkl-2019.5-075

# Add the path for libiomp5.so to make sure this lib is able to be found.
export LD_LIBRARY_PATH=/opt/intel/lib/intel64_lin:$LD_LIBRARY_PATH
```

```bash
# Install MXNet
git clone https://github.com/apache/incubator-mxnet.git
cd incubator-mxnet
git checkout 6ae469a17ebe517325cdf6acdf0e2a8b4d464734
git submodule update --init
make -j USE_OPENCV=1 USE_MKLDNN=1 USE_BLAS=mkl USE_PROFILER=0 USE_LAPACK=0 USE_GPERFTOOLS=0 USE_INTEL_PATH=/opt/intel/
cd python && python setup.py install
cd ../

# Install ONNX
conda install -c conda-forge protobuf=3.9 onnx
pip install opencv-python pycocotools onnxruntim
```

### 2.2 Convert Model from ONNX to MXNet
Convert the ONNX model to MXNet by [onnx2mxnet.py](../../code/resnet/resnet-mx/tools/onnx2mxnet.py)

```bash
mkdir model
# Download ONNX model
wget -O ./model/resnet50-v1.5.onnx https://zenodo.org/record/2592612/files/resnet50_v1.onnx
# Convert to MXNet
python tools/onnx2mxnet.py
```
The converted FP32 model for MXNet is located at: `model/resnet50_v1b-symbol.json` and `model/resnet50_v1b-0000.params`.

## 3. Quantize FP32 model to INT8 model
To get INT8 model you would need to quantize the model using calibration dataset and quantization tool.

### 3.1 Prepare Calibration Dataset
The calibration dataset (image list) is from [mlperf](http://github.com/mlperf/inference/blob/master/calibration/ImageNet/cal_image_list_option_1.txt).

### 3.2 Quantization Tool Installation
Intel® Low Precision Optimization Tool is used to quantize the FP32 model, refer [here](https://github.com/intel/lp-opt-tool) for more detail information.

Follow the instructions to install Intel® Low Precision Optimization Tool:
```bash
git clone https://github.com/intel/lp-opt-tool
cp ilit_calib.patch lp-opt-tool/
cd lp-opt-tool && git checkout c468259 && git apply ilit_calib.patch
python setup.py install
cd ..
```

### 3.3 Quantize Model With Calibration Dataset
Quantize and calibrate the model by [calib.sh](../../code/resnet/resnet-mx/calib.sh)

```bash
# update the following path based on your env
export DATASET_PATH=/lustre/dataset/imagenet/img_raw/ILSVRC2012_img_val
export DATASET_LIST=./val_map.txt
export CALIBRATION_IMAGE_LIST=./cal_image_list_option_1.txt
./calib.sh
```
After quantization, the INT8 model is located at: `model/resnet50_v1b-quantized-symbol.json` and `model/resnet50_v1b-quantized-0000.params`.
