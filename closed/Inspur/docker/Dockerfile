# Copyright (c) 2018-2021, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Explicitly use bash instead of sh ('echo' behaves differently on some shells)
SHELL ["/bin/bash", "-c"]

ARG CUDA_VER=11.1
ARG DRIVER_VER_MAJOR=455
ARG SM_GENCODE=75
ARG USE_CPU=0

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential autoconf libtool git \
        ccache curl wget pkg-config sudo ca-certificates vim-tiny automake libssl-dev \
        bc python3-dev python3-pip python2.7-dev google-perftools gdb libglib2.0-dev clang python-pip sshfs libre2-dev \
        libboost-dev libnuma-dev numactl sysstat sshpass \
 && apt-get remove -y cmake \
 && apt-get remove -y libgflags-dev \
 && apt-get remove -y libprotobuf-dev \
 && apt-get -y autoremove \
 && rm -rf /var/lib/apt/lists/*

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

WORKDIR /tmp

# Needed by LoadGen
RUN python2 -m pip install --upgrade setuptools \
 && python2 -m pip install wheel==0.35.1 \
 && python2 -m pip install absl-py==0.7.1

RUN python3 -m pip install --upgrade pip \
 && python3 -m pip install --upgrade setuptools wheel virtualenv \
 && python3 -m pip install cmake==3.18.4 \
 && python3 -m pip install pillow==6.0.0 \
 && python3 -m pip install opencv-python-headless \
 && python3 -m pip install openpyxl==3.0.4 \
 && python3 -m pip install pyyaml==5.1.2

# Needed for power measurements
RUN python3 -m pip install pyvisa pyvisa-py

# install gflags
# -DBUILD_SHARED_LIBS=ON -DBUILD_STATIC_LIBS=ON -DBUILD_gflags_LIB=ON .. \
RUN git clone -b v2.2.1 https://github.com/gflags/gflags.git \
 && cd gflags \
 && mkdir build && cd build \
 && cmake -DBUILD_SHARED_LIBS=ON -DBUILD_STATIC_LIBS=ON -DBUILD_gflags_LIB=ON .. \
 && make -j \
 && make install \
 && cd /tmp && rm -rf gflags

# install glog
RUN git clone -b v0.3.5 https://github.com/google/glog.git \
 && cd glog \
 && cmake -H. -Bbuild -G "Unix Makefiles" -DBUILD_SHARED_LIBS=ON -DBUILD_STATIC_LIBS=ON \
 && cmake --build build \
 && cmake --build build --target install \
 && cd /tmp && rm -rf glog

RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config zip g++ zlib1g-dev unzip python \
 && rm -rf /var/lib/apt/lists/*

# Install TensorFlow, needed by SSDMobileNet benchmark
# Install CPU version since we don't actually need to run TensorFlow.
RUN python3 -m pip install tensorflow==1.13.1

# Install pytorch and torchvision CPU version since the GPU version causes cudnn linking issue
RUN python3 -m pip install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html \
 && python3 -m pip install onnx==1.6.0 \
 && python3 -m pip install pycuda==2019.1.2 \
 && python3 -m pip install Cython==0.29.10 \
 && python3 -m pip install pycocotools==2.0.0 \
 && python3 -m pip install scikit-learn==0.22.2 \
 && python3 -m pip install numpy==1.16.0 \
 && python3 -m pip install pytest==4.6.2 \
 && python3 -m pip install matplotlib==3.0.2

# Install CUB, needed by NMS OPT plugin
RUN wget https://github.com/NVlabs/cub/archive/1.8.0.zip -O cub-1.8.0.zip \
 && unzip cub-1.8.0.zip \
 && mv cub-1.8.0/cub /usr/include/x86_64-linux-gnu/ \
 && rm -rf cub-1.8.0.zip cub-1.8.0

# Install libjemalloc2
RUN echo 'deb http://archive.ubuntu.com/ubuntu focal main restricted universe multiverse' | tee -a /etc/apt/sources.list.d/focal.list \
  && echo 'Package: *\nPin: release a=focal\nPin-Priority: -10\n' | tee -a /etc/apt/preferences.d/focal.pref \
  && apt-get update \
  && apt-get install --no-install-recommends -t focal -y libjemalloc2 libtcmalloc-minimal4

# Install HuggingFace Transformers
RUN python3 -m pip install transformers==2.4.0

# Install nnUnet, needed by 3D-Unet.
RUN cd /tmp \
 && git clone https://github.com/MIC-DKFZ/nnUNet.git \
 && cd nnUNet \
 && git checkout cf28f1f223208db0df91e1b4354ad26145d97a73 \
 && python3 -m pip install . \
 && cd /tmp \
 && rm -rf nnUNet

# Install dependencies needed for RNN-T preprocessing
RUN apt-get install -y sox \
 && python3 -m pip install sox==1.4.0 toml==0.10.1 unidecode==1.1.1 inflect==4.1.0 librosa==0.8.0 \
 && cd /tmp \
 && git clone https://www.github.com/nvidia/apex \
 && cd apex \
 && python3 setup.py install \
 && cd /tmp \
 && rm -rf apex

# Needed by official RNNT accuracy script
RUN apt-get install -y --no-install-recommends libsndfile1-dev

# Install rapidJSON, needed by Triton
RUN apt-get update && apt-get install rapidjson-dev

# Install DALI 0.30.0
RUN python3 -m pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda110==0.30

# Install CUDNN 8.1.1
# For some reason, installing this after TRT causes TRT to be uninstalled by apt.
RUN apt-get update \
 && apt-get install -y --allow-change-held-packages libcudnn8=8.1.1.33-1+cuda11.2 \
 && apt-get install -y --allow-change-held-packages libcudnn8-dev=8.1.1.33-1+cuda11.2

# Install TRT 7.2
ARG NV_DEB_URL=https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64
RUN cd /tmp \
    && install_deb_pkg() { wget $NV_DEB_URL/$1 -O $1 && dpkg -i $1 && rm $1; } \
    && install_deb_pkg libnvinfer7_7.2.3-1+cuda11.1_amd64.deb \
    && install_deb_pkg libnvinfer-plugin7_7.2.3-1+cuda11.1_amd64.deb \
    && install_deb_pkg libnvparsers7_7.2.3-1+cuda11.1_amd64.deb \
    && install_deb_pkg libnvonnxparsers7_7.2.3-1+cuda11.1_amd64.deb \
    && install_deb_pkg libnvinfer-dev_7.2.3-1+cuda11.1_amd64.deb \
    && install_deb_pkg libnvinfer-plugin-dev_7.2.3-1+cuda11.1_amd64.deb \
    && install_deb_pkg libnvparsers-dev_7.2.3-1+cuda11.1_amd64.deb \
    && install_deb_pkg libnvonnxparsers-dev_7.2.3-1+cuda11.1_amd64.deb \
    && install_deb_pkg python3-libnvinfer_7.2.3-1+cuda11.1_amd64.deb \
    && install_deb_pkg python3-libnvinfer-dev_7.2.3-1+cuda11.1_amd64.deb \
    && unset -f install_deb_pkg

# With latest Ubuntu:18.04 container, there will be no 'python' or 'pip' even if we have installed 'python3' and
# 'python3-pip'. So add softlink to avoid wheel installation failure.
RUN ln -sf /usr/bin/python3 /usr/bin/python
RUN ln -sf /usr/bin/pip3 /usr/bin/pip
RUN python3 -m pip install nvidia-pyindex \
    && python3 -m pip install uff graphsurgeon onnx-graphsurgeon

# Explicitly downgrade numpy since cocoeval requires numpy 1.16.x.
RUN python3 -m pip install numpy==1.16.0

# Power-dev needs Python 3.7, but we still want default python3 to be 3.6.
RUN apt install -y python3.7 \
 && rm -rf /usr/bin/python3 \
 && ln -s /usr/bin/python3.6 /usr/bin/python3

# Update GLIBC version needed for Triton TF-CPU backend
RUN if [ ${USE_CPU} = 1 ]; then \
 apt-get upgrade -y libstdc++6; fi

ENV USE_CPU=${USE_CPU}

WORKDIR /work
