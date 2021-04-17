#!/bin/bash
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

sudo apt-get install -y cuda-toolkit-10.2
sudo apt-get install -y python-dev python3-dev python-pip python3-pip
sudo apt-get install -y virtualenv moreutils libnuma-dev numactl sshpass
sudo apt-get install -y cmake pkg-config zip g++ unzip zlib1g-dev
sudo apt-get install -y --no-install-recommends clang libglib2.0-dev
sudo apt-get install -y libhdf5-serial-dev hdf5-tools libopenmpi2
sudo apt-get install -y zlib1g-dev zip libjpeg8-dev libhdf5-dev libtiff5-dev

# matplotlib dependencies
sudo apt-get install -y libssl-dev libfreetype6-dev libpng-dev

sudo apt-get install -y libatlas3-base libopenblas-base
sudo apt-get install -y git
sudo apt-get install -y git-lfs && git-lfs install

# CMake assumes that the CUDA toolkit is located in /usr/local/cuda
if [[ -e /usr/local/cuda/packages ]]; then sudo mv /usr/local/cuda /usr/local/cuda_packages; fi
sudo ln -s /usr/local/cuda-10.2 /usr/local/cuda

cd /tmp

# install cub
wget https://github.com/NVlabs/cub/archive/1.8.0.zip -O cub-1.8.0.zip \
 && unzip cub-1.8.0.zip \
 && sudo mv cub-1.8.0/cub /usr/include/aarch64-linux-gnu/ \
 && rm -rf cub-1.8.0.zip cub-1.8.0

# install gflags
sudo rm -rf gflags \
 && git clone -b v2.2.1 https://github.com/gflags/gflags.git \
 && cd gflags \
 && mkdir build && cd build \
 && cmake -DBUILD_SHARED_LIBS=ON -DBUILD_STATIC_LIBS=ON -DBUILD_gflags_LIB=ON .. \
 && make -j \
 && sudo make install \
 && cd /tmp && rm -rf gflags

# install glog
sudo rm -rf grpc \
 && git clone -b v0.3.5 https://github.com/google/glog.git \
 && cd glog \
 && cmake -H. -Bbuild -G "Unix Makefiles" -DBUILD_SHARED_LIBS=ON -DBUILD_STATIC_LIBS=ON \
 && cmake --build build \
 && sudo cmake --build build --target install \
 && cd /tmp && rm -rf glog

# Install other dependencies (PyTorch, TensorFlow, etc.)
export CUDA_ROOT=/usr/local/cuda-10.2
export CUDA_INC_DIR=$CUDA_ROOT/include
export PATH=$CUDA_ROOT/bin:$PATH
export CPATH=$CUDA_ROOT/include:$CPATH
export LIBRARY_PATH=$CUDA_ROOT/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_ROOT/lib64:$LD_LIBRARY_PATH
sudo python3 -m pip install -U --index-url https://pypi.org/simple --no-cache-dir setuptools==41.0.1 \
 && sudo python3 -m pip install Cython==0.29.10 \
 && sudo python3 -m pip install numpy==1.16.4 \
 && sudo python3 -m pip install matplotlib==3.0.2 \
 && sudo python3 -m pip install grpcio==1.16.1 \
 && sudo python3 -m pip install absl-py==0.7.1 \
 && sudo python3 -m pip install py-cpuinfo==5.0.0 \
 && sudo python3 -m pip install portpicker==1.3.1 \
 && sudo python3 -m pip install grpcio==1.16.1 \
 && sudo python3 -m pip install six==1.12.0 \
 && sudo python3 -m pip install mock==3.0.5 \
 && sudo python3 -m pip install requests==2.22.0 \
 && sudo python3 -m pip install gast==0.2.2 \
 && sudo python3 -m pip install h5py==2.10.0 \
 && sudo python3 -m pip install astor==0.8.0 \
 && sudo python3 -m pip install termcolor==1.1.0 \
 && sudo python3 -m pip install pytest==5.1.2 \
 && sudo python3 -m pip install pillow==6.0.0 \
 && sudo python3 -m pip install scikit-learn==0.23.0 \
 && sudo python3 -m pip install pyvisa-py==0.5.1 \
 && sudo python3 -m pip install protobuf==3.6.1 \
 && sudo python3 -m pip install keras-preprocessing==1.0.5 \
 && sudo python3 -m pip install tensorflow-estimator==1.15.1 \
 && sudo python3 -m pip install tensorboard==1.15.0 \
 && sudo python3 -m pip install keras-applications==1.0.8 \
 && sudo python3 -m pip install wrapt==1.12.1 \
 && sudo python3 -m pip install google-pasta==0.1.6 \
 && sudo python3 -m pip install opt-einsum==2.3.2 \
 && sudo python3 -m pip install --no-deps --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v45 'tensorflow<2' \
 && wget -O torch-1.4.0-cp36-cp36m-linux_aarch64.whl https://nvidia.box.com/shared/static/c3d7vm4gcs9m728j6o5vjay2jdedqb55.whl \
 && sudo python3 -m pip install torch-1.4.0-cp36-cp36m-linux_aarch64.whl \
 && rm torch-1.4.0-cp36-cp36m-linux_aarch64.whl \
 && sudo python3 -m pip install torchvision==0.2.2.post3 \
 && sudo -E python3 -m pip install pycuda==2019.1.2 \
 && sudo python3 -m pip install pycocotools==2.0.0 \
 && sudo python -m pip install absl-py==0.7.1 \
 && sudo python3 -m pip install numpy==1.16.4 \
 && sudo python3 -m pip install tqdm==4.46.0 \
 && sudo python3 -m pip install onnx==1.7.0

# Install DALI 0.25.0, needed by RNN-T
sudo python3 -m pip install protobuf==3.11.1 \
 && wget https://github.com/protocolbuffers/protobuf/releases/download/v3.11.1/protobuf-cpp-3.11.1.tar.gz \
 && tar -xzf protobuf-cpp-3.11.1.tar.gz \
 && rm protobuf-cpp-3.11.1.tar.gz \
 && cd protobuf-3.11.1 \
 && ./configure CXXFLAGS="-fPIC" --prefix=/usr/local --disable-shared \
 && make -j8 \
 && sudo make install \
 && sudo ldconfig \
 && cd /tmp \
 && rm -rf protobuf-3.11.1 \
 && sudo apt purge -y cmake \
 && wget https://github.com/Kitware/CMake/releases/download/v3.17.3/cmake-3.17.3.tar.gz \
 && tar -xzf cmake-3.17.3.tar.gz \
 && rm cmake-3.17.3.tar.gz \
 && cd cmake-3.17.3 \
 && ./bootstrap --prefix=/usr -- -DCMAKE_BUILD_TYPE:STRING=Release \
 && make -j8 \
 && sudo make install \
 && cd /tmp \
 && rm -rf cmake-3.17.3 \
 && cd /usr/local \
 && sudo git clone -b release_v0.25 --recursive https://github.com/NVIDIA/DALI \
 && cd DALI \
 && sudo mkdir build \
 && cd build \
 && sudo cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCUDA_TARGET_ARCHS="72" \
    -DBUILD_PYTHON=ON -DBUILD_TEST=OFF -DBUILD_BENCHMARK=OFF -DBUILD_LMDB=OFF -DBUILD_NVTX=OFF -DBUILD_NVJPEG=OFF \
    -DBUILD_LIBTIFF=OFF -DBUILD_NVOF=OFF -DBUILD_NVDEC=OFF -DBUILD_LIBSND=OFF -DBUILD_NVML=OFF -DBUILD_FFTS=ON \
    -DVERBOSE_LOGS=OFF -DWERROR=OFF -DBUILD_WITH_ASAN=OFF .. \
 && sudo make -j8 \
 && sudo make install \
 && sudo python3 -m pip install dali/python/ \
 && sudo mv /usr/local/DALI/build/dali/python/nvidia/dali /tmp/dali \
 && sudo rm -rf /usr/local/DALI \
 && sudo mkdir -p /usr/local/DALI/build/dali/python/nvidia/ \
 && sudo mv /tmp/dali /usr/local/DALI/build/dali/python/nvidia/ \
 && cd /tmp

# Install ONNX graph surgeon, needed for 3D-Unet ONNX preprocessing.
cd /tmp \
 && git clone https://github.com/NVIDIA/TensorRT.git \
 && cd TensorRT \
 && git checkout release/7.1 \
 && cd tools/onnx-graphsurgeon \
 && make build \
 && sudo python3 -m pip install --no-deps -t /usr/local/lib/python3.6/dist-packages --force-reinstall dist/*.whl \
 && cd /tmp \
 && rm -rf TensorRT

# Explicitly downgrade numpy since cocoeval requires numpy 1.16.x.
sudo python3 -m pip install numpy==1.16.4

# Power-dev needs Python 3.7, but we still want default python3 to be 3.6.
sudo apt install -y python3.7 \
 && sudo rm -rf /usr/bin/python3 \
 && sudo ln -s /usr/bin/python3.6 /usr/bin/python3
