# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

SHELL := /bin/bash
MAKEFILE_NAME := $(lastword $(MAKEFILE_LIST))

ARCH := $(shell uname -p)
UNAME := $(shell whoami)
UID := $(shell id -u `whoami`)
HOSTNAME := $(shell hostname)
GROUPNAME := $(shell id -gn `whoami`)
GROUPID := $(shell id -g `whoami`)
TIMESTAMP := $(shell date +'%Y.%m.%d-%H.%M.%S')
ifndef HOST_HOSTNAME
    HOST_HOSTNAME := $(HOSTNAME)
endif
HOST_HOSTNAME_SHORT := $(firstword $(subst ., , $(HOST_HOSTNAME)))
HOSTNAME_SHORT := $(firstword $(subst ., , $(HOSTNAME)))

USE_CPU ?= 0
SUBMITTER ?= NVIDIA

# Conditional Docker flags
ifndef DOCKER_DETACH
    DOCKER_DETACH := 0
endif
ifndef DOCKER_TAG
    DOCKER_TAG := $(UNAME)
endif
DOCKER_BUILDKIT ?= 1

DOCKER_NAME := mlperf-inference-$(DOCKER_TAG)

PROJECT_ROOT := $(shell pwd)
BUILD_DIR    := $(PROJECT_ROOT)/build

HOST_VOL ?= ${PWD}
CONTAINER_VOL ?= /work
NO_DOCKER_PULL ?= 0
NO_BUILD ?= 0

# nsys and nvprof locked clock frequency
GPUCLK?=1000

# Set the include directory for Loadgen header files
INFERENCE_DIR = $(BUILD_DIR)/inference
INFERENCE_URL = https://github.com/mlcommons/inference.git
LOADGEN_INCLUDE_DIR := $(INFERENCE_DIR)/loadgen
LOADGEN_LIB_DIR := $(LOADGEN_INCLUDE_DIR)/build
INFERENCE_HASH = 8ab63193d821dfa3adc78054d5831ea76a9ed502

# Set the power-dev directory
POWER_DEV_DIR = $(BUILD_DIR)/power-dev
POWER_DEV_URL = https://github.com/mlcommons/power-dev.git
POWER_DEV_HASH = 4d8e5e9d57e6b15813bd240ae3d17fad185a2fcc
POWER_CLIENT_SCRIPT = $(BUILD_DIR)/power-dev/ptd_client_server/client.py
POWER_NTP_URL = ib-01.dc4-in.nvidia.com
ifeq ($(HOST_HOSTNAME_SHORT), computelab-ro-prod-01)
    POWER_SERVER_IP = 10.117.16.94
else ifeq ($(HOST_HOSTNAME_SHORT), ipp1-1469)
    POWER_SERVER_IP = 10.117.16.11
else ifeq ($(HOST_HOSTNAME_SHORT), sjc1-luna-02)
    POWER_SERVER_IP = 10.33.72.241
else ifeq ($(HOST_HOSTNAME_SHORT), computelab-310)
    POWER_SERVER_IP = 10.33.72.216
else ifeq ($(HOST_HOSTNAME_SHORT), computelab-501)
    POWER_SERVER_IP = 10.117.16.63
endif

POWER_SERVER_USERNAME = lab
POWER_SERVER_PASSWORD = labuser
POWER_SERVER_USER_IP = $(POWER_SERVER_USERNAME)@$(POWER_SERVER_IP)
POWER_SERVER_CONFIG = power/server-$(HOST_HOSTNAME_SHORT).cfg
POWER_SERVER_SCRIPT_DIR = C:/power-dev/ptd_client_server
POWER_SERVER_LOG_DIR = C:/ptd-logs
POWER_SSH = sshpass -p $(POWER_SERVER_PASSWORD) ssh -o "StrictHostKeyChecking no"
POWER_SCP = sshpass -p $(POWER_SERVER_PASSWORD) scp -o "StrictHostKeyChecking no"

# Set log directories for power logs
POWER_LOGS_TEMP_DIR = $(BUILD_DIR)/power_logs_temp
POWER_LOGS_DIR = $(BUILD_DIR)/power_logs

# Set the Triton directory
TRITON_DIR = $(BUILD_DIR)/triton-inference-server
TRITON_OUT_DIR = $(TRITON_DIR)/out
TRITON_URL = https://github.com/triton-inference-server/server
TRITON_HASH = mlperf-inference-v1.0

# Set Environment variables to extracted contents
export LD_LIBRARY_PATH := $(LD_LIBRARY_PATH):/usr/local/cuda/lib64:/usr/lib/$(ARCH)-linux-gnu:$(LOADGEN_LIB_DIR)
export LIBRARY_PATH := /usr/local/cuda/lib64:/usr/lib/$(ARCH)-linux-gnu:$(LOADGEN_LIB_DIR):$(LIBRARY_PATH)
export PATH := /usr/local/cuda/bin:$(PATH)
export CPATH := /usr/local/cuda/include:/usr/include/$(ARCH)-linux-gnu:/usr/include/$(ARCH)-linux-gnu/cub:$(CPATH)
export CUDA_PATH := /usr/local/cuda
export PYTHONPATH := $(LOADGEN_LIB_DIR):$(PYTHONPATH)
export CCACHE_DISABLE=1
export NUMBA_CACHE_DIR=$(BUILD_DIR)/cache

# Set CUDA_DEVICE_MAX_CONNECTIONS to increase multi-stream performance.
export CUDA_DEVICE_MAX_CONNECTIONS := 32

# Set DATA_DIR, PREPROCESSED_DATA_DIR, and MODEL_DIR if they are not already set
ifndef DATA_DIR
    export DATA_DIR := $(BUILD_DIR)/data
endif
ifndef PREPROCESSED_DATA_DIR
    export PREPROCESSED_DATA_DIR := $(BUILD_DIR)/preprocessed_data
endif
ifndef MODEL_DIR
    export MODEL_DIR := $(BUILD_DIR)/models
endif

# Please run `export MLPERF_SCRATCH_PATH=<path>` to set your scratch space path.
# The below paths are for internal use only.
ifneq ($(wildcard /home/scratch.mlperf_inference),)
    MLPERF_SCRATCH_PATH ?= /home/scratch.mlperf_inference
endif
# Please run `export MLPERF_CPU_SCRATCH_PATH=<path>` to set your scratch space path.
# The below paths are for internal use only.
ifneq ($(wildcard /home/scratch.mlperf_inference_triton_cpu_data),)
	MLPERF_CPU_SCRATCH_PATH ?= /home/scratch.mlperf_inference_triton_cpu_data
endif
ifneq ($(wildcard /home/scratch.svc_compute_arch),)
    DOCKER_MOUNTS += -v /home/scratch.svc_compute_arch:/home/scratch.svc_compute_arch
endif
ifneq ($(wildcard /home/scratch.dlsim),)
    DOCKER_MOUNTS += -v /home/scratch.dlsim:/home/scratch.dlsim
endif
ifneq ($(wildcard $(PROJECT_ROOT)/../../regression),)
    DOCKER_MOUNTS += -v $(PROJECT_ROOT)/../../regression:/regression
endif
ifdef MLPERF_SCRATCH_PATH
    ifneq ($(wildcard $(MLPERF_SCRATCH_PATH)),)
        DOCKER_MOUNTS += -v $(MLPERF_SCRATCH_PATH):$(MLPERF_SCRATCH_PATH)
    endif
endif
ifdef MLPERF_CPU_SCRATCH_PATH
    ifneq ($(wildcard $(MLPERF_CPU_SCRATCH_PATH)),)
        DOCKER_MOUNTS += -v $(MLPERF_CPU_SCRATCH_PATH):$(MLPERF_CPU_SCRATCH_PATH)
    endif
endif

# Specify default dir for harness output logs.
ifndef LOG_DIR
    export LOG_DIR := $(BUILD_DIR)/logs/$(TIMESTAMP)
endif

# Specify debug options for build (default to Release build)
ifeq ($(DEBUG),1)
    BUILD_TYPE := Debug
else
    BUILD_TYPE := Release
endif

# Handle different nvidia-docker version. Do not use nvidia-docker when running with CPUs
ifeq ($(USE_CPU), 1)
	DOCKER_RUN_CMD := docker run
else
ifneq ($(wildcard /usr/bin/nvidia-docker),)
    DOCKER_RUN_CMD := nvidia-docker run
    # Set Environment variables to fix docker client and server version mismatch
    # Related issue: https://github.com/kubernetes-sigs/kubespray/issues/6160
    export DOCKER_API_VERSION=1.40
else
    DOCKER_RUN_CMD := docker run --gpus=all
endif
endif

# If DOCKER_COMMAND is not pass, launch interactive docker container session.
ifeq ($(DOCKER_COMMAND),)
	DOCKER_INTERACTIVE_FLAGS = -it
else
	DOCKER_INTERACTIVE_FLAGS =
endif

ALLOWED_MIG_CONFS := OFF 1 2 3 ALL
MIG_CONF ?= OFF
ifeq ($(filter $(MIG_CONF),$(ALLOWED_MIG_CONFS)),)
    $(warning MIG_CONF was not set to a valid value. Default to OFF to be safe.)
    MIG_CONF := OFF
endif

ifneq ($(ARCH), aarch64)
    ifneq ($(USE_CPU), 1)
        ifneq ($(SKIP_DRIVER_CHECK), 1)
            DRIVER_VER_MAJOR ?= $(shell nvidia-smi | /bin/grep -Eo 'Driver Version: [+-]?[0-9]+' | awk -F ' ' '{print $$NF}')

            # Check driver version and launch the appropriate container. If driver >= 455, use CUDA 11.1, otherwise 11.0
            ifeq ($(shell if [ $(DRIVER_VER_MAJOR) -ge 455 ]; then echo true; else echo false; fi), true)
                CUDA_VER := 11.1
                # Workaround: Some machines were upgraded to driver 460, but TRT build is listed under r455, so it
                # will fail to find the URL
                DRIVER_VER_MAJOR := 455
            else
                $(error MLPerf Inference v1.0 code requires NVIDIA Driver Version >= 455.xx)
            endif # Driver check
        else
            $(warning Assumes system has NVIDIA Driver Version >= 455.xx)
            CUDA_VER := 11.1
            DRIVER_VER_MAJOR := 455
        endif # SKIP_DRIVER_CHECK check
    else
        CUDA_VER := 11.1
        DRIVER_VER_MAJOR := 455
    endif # triton cpu check

    DOCKER_IMAGE_NAME := base-cuda$(CUDA_VER)
    # Check if we are on intranet
    ifeq ($(shell bash $(PROJECT_ROOT)/scripts/check_intranet.sh),0)
        BASE_IMAGE ?= gitlab-master.nvidia.com/compute/mlperf-inference:$(DOCKER_IMAGE_NAME)
    else
        BASE_IMAGE ?= nvidia/cuda:$(CUDA_VER)-cudnn8-devel-ubuntu18.04
    endif # check_intranet
endif # aarch64 check

SM_GENCODE := $(shell ./scripts/get_gencode_$(ARCH) 2> /dev/null)

############################## PREBUILD ##############################
# Build the docker image and launch an interactive container.
# For CPU builds, first build the backend libraries and copy them into the working directory
.PHONY: prebuild
prebuild:
ifeq ($(USE_CPU), 1)
	@$(MAKE) -f $(MAKEFILE_NAME) build_triton_cpu_backends
endif
	@$(MAKE) -f $(MAKEFILE_NAME) build_docker NO_BUILD?=1
ifneq ($(strip ${DOCKER_DETACH}), 1)
	@$(MAKE) -f $(MAKEFILE_NAME) configure_mig MIG_CONF=$(MIG_CONF)
	@$(MAKE) -f $(MAKEFILE_NAME) attach_docker || true
	@$(MAKE) -f $(MAKEFILE_NAME) teardown_mig MIG_CONF=$(MIG_CONF)
endif

# Configure MIG
.PHONY: configure_mig
configure_mig:
	if [ $(MIG_CONF) != "OFF" ]; then MIG_CONF=$(MIG_CONF) ./scripts/mig_configure.sh; fi

# Tear down MIG
.PHONY: teardown_mig
teardown_mig:
	if [ $(MIG_CONF) != "OFF" ]; then ./scripts/mig_teardown.sh; fi

# Clone Triton.
.PHONY: clone_triton
clone_triton:
ifeq ($(ARCH), x86_64)
	@if [ ! -d $(TRITON_DIR) ]; then \
		echo "Cloning Triton Inference Server" \
			&& git clone $(TRITON_URL) $(TRITON_DIR); \
	fi
	@cd $(TRITON_DIR) && git fetch && git checkout $(TRITON_HASH)
endif

# Add symbolic links to scratch path if it exists.
.PHONY: link_dirs
link_dirs:
	@mkdir -p build
ifeq ($(USE_CPU), 1)
ifdef MLPERF_CPU_SCRATCH_PATH
	@mkdir -p $(MLPERF_CPU_SCRATCH_PATH)/data
	@mkdir -p $(MLPERF_CPU_SCRATCH_PATH)/preprocessed_data
	@mkdir -p $(MLPERF_CPU_SCRATCH_PATH)/models
	@ln -sfn $(MLPERF_CPU_SCRATCH_PATH)/backends/openvino_v2 openvino
	@ln -sfn $(MLPERF_CPU_SCRATCH_PATH)/data $(DATA_DIR)
	@ln -sfn $(MLPERF_CPU_SCRATCH_PATH)/preprocessed_data $(PREPROCESSED_DATA_DIR)
	@ln -sfn $(MLPERF_CPU_SCRATCH_PATH)/models/Triton $(MODEL_DIR)
endif
else
ifdef MLPERF_SCRATCH_PATH
	@mkdir -p $(MLPERF_SCRATCH_PATH)/data
	@mkdir -p $(MLPERF_SCRATCH_PATH)/preprocessed_data
	@mkdir -p $(MLPERF_SCRATCH_PATH)/models
	@ln -sfn $(MLPERF_SCRATCH_PATH)/data $(DATA_DIR)
	@ln -sfn $(MLPERF_SCRATCH_PATH)/preprocessed_data $(PREPROCESSED_DATA_DIR)
	@ln -sfn $(MLPERF_SCRATCH_PATH)/models $(MODEL_DIR)
endif
endif

# Small helper to check if nvidia-docker is installed correctly.
.PHONY: docker_sanity
docker_sanity:
	docker pull nvcr.io/nvidia/cuda:11.0.3-runtime-ubuntu18.04
	$(DOCKER_RUN_CMD) --rm \
		-e NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES} \
		nvcr.io/nvidia/cuda:11.0.3-runtime-ubuntu18.04 nvidia-smi
	@echo "Nvidia-docker is installed correctly!"

# Build the docker image.
.PHONY: build_docker
build_docker:
ifeq ($(ARCH), x86_64)
	@echo "Building Docker image"
ifeq ($(NO_DOCKER_PULL), 0)
ifneq ($(USE_CPU), 1)
	docker pull $(BASE_IMAGE)
endif
endif
	DOCKER_BUILDKIT=$(DOCKER_BUILDKIT) docker build -t mlperf-inference:$(DOCKER_TAG)-latest \
		--build-arg BASE_IMAGE=$(BASE_IMAGE) \
		--build-arg CUDA_VER=$(CUDA_VER) \
		--build-arg DRIVER_VER_MAJOR=$(DRIVER_VER_MAJOR) \
		--build-arg SM_GENCODE=$(SM_GENCODE) \
		--build-arg USE_CPU=$(USE_CPU) \
		--network host - < docker/Dockerfile
ifeq ($(NO_BUILD), 0)
	DOCKER_BUILDKIT=$(DOCKER_BUILDKIT) docker build -t mlperf-inference:$(DOCKER_TAG)-latest --no-cache --network host \
		--build-arg BASE_IMAGE=mlperf-inference:$(DOCKER_TAG)-latest \
		-f docker/Dockerfile.build .
endif
endif

# Add current user into docker image.
.PHONY: docker_add_user
docker_add_user:
ifeq ($(ARCH), x86_64)
	@echo "Adding user account into image"
	DOCKER_BUILDKIT=$(DOCKER_BUILDKIT) docker build -t mlperf-inference:$(DOCKER_TAG) --network host \
		--build-arg BASE_IMAGE=mlperf-inference:$(DOCKER_TAG)-latest \
		--build-arg GID=$(GROUPID) --build-arg UID=$(UID) --build-arg GROUP=$(GROUPNAME) --build-arg USER=$(UNAME) \
		- < docker/Dockerfile.user
endif

# Add user and launch an interactive container session.
.PHONY: attach_docker
attach_docker:
	@$(MAKE) -f $(MAKEFILE_NAME) docker_add_user
	@$(MAKE) -f $(MAKEFILE_NAME) launch_docker

# Launch a container session.
.PHONY: launch_docker
launch_docker:
ifeq ($(ARCH), x86_64)
	@echo "Launching Docker session"
	$(DOCKER_RUN_CMD) --rm $(DOCKER_INTERACTIVE_FLAGS) -w /work \
		-v $(HOST_VOL):$(CONTAINER_VOL) -v ${HOME}:/mnt/${HOME} \
		--cap-add SYS_ADMIN --cap-add SYS_TIME \
		-e NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES} \
		--shm-size=32gb \
		-v /etc/timezone:/etc/timezone:ro -v /etc/localtime:/etc/localtime:ro \
		--security-opt apparmor=unconfined --security-opt seccomp=unconfined \
		--name $(DOCKER_NAME) -h $(DOCKER_NAME) --add-host $(DOCKER_NAME):127.0.0.1 \
		--user $(UID):$(GROUPID) --net host --device /dev/fuse \
		$(DOCKER_MOUNTS) $(DOCKER_ARGS) \
		-e MLPERF_SCRATCH_PATH=$(MLPERF_SCRATCH_PATH) \
		-e HOST_HOSTNAME=$(HOSTNAME) \
		$(shell if [ $(MIG_CONF) == "ALL" ]; then echo "--gpus all -e NVIDIA_MIG_CONFIG_DEVICES=all"; elif [ $(MIG_CONF) != "OFF" ]; then echo "--gpus '\"device=`bash scripts/mig_get_uuid.sh`\"'"; fi) \
		mlperf-inference:$(DOCKER_TAG) $(DOCKER_COMMAND)
endif

############################## DOWNLOAD_MODEL ##############################

BENCHMARKS = resnet50 ssd-resnet34 ssd-mobilenet bert dlrm rnnt 3d-unet

.PHONY: download_model
download_model: link_dirs
	$(foreach _benchmark,$(BENCHMARKS),bash code/$(_benchmark)/tensorrt/download_model.sh &&) \
		echo "Finished downloading all the models!"

############################## DOWNLOAD_DATA ##############################

.PHONY: download_data
download_data: link_dirs
	@$(foreach _benchmark,$(BENCHMARKS),bash code/$(_benchmark)/tensorrt/download_data.sh &&) \
		echo "Finished downloading all the datasets!"

############################## PREPROCESS_DATA ##############################

.PHONY: preprocess_data
preprocess_data: link_dirs
	@$(foreach _benchmark,$(BENCHMARKS),python3 code/$(_benchmark)/tensorrt/preprocess_data.py --data_dir=$(DATA_DIR) --preprocessed_data_dir=$(PREPROCESSED_DATA_DIR) &&) \
		echo "Finished preprocessing all the datasets!"

############################### BUILD GPU ###############################

# Build all source codes.
.PHONY: build
build: clone_loadgen clone_power_dev clone_triton link_dirs
	@$(MAKE) -f $(MAKEFILE_NAME) build_triton
	@$(MAKE) -f $(MAKEFILE_NAME) build_plugins
	@$(MAKE) -f $(MAKEFILE_NAME) build_loadgen
	@$(MAKE) -f $(MAKEFILE_NAME) build_harness

# Clone LoadGen repo.
.PHONY: clone_loadgen
clone_loadgen:
	@if [ ! -d $(LOADGEN_INCLUDE_DIR) ]; then \
		echo "Cloning Official MLPerf Inference (For Loadgen Files)" \
			&& git clone $(INFERENCE_URL) $(INFERENCE_DIR); \
	fi
	@echo "Updating Loadgen" \
		&& cd $(INFERENCE_DIR) \
		&& git fetch \
		&& git checkout $(INFERENCE_HASH) \
		&& git submodule update --init tools/submission/power-dev \
		&& git submodule update --init third_party/pybind \
		&& git submodule update --init language/bert/DeepLearningExamples \
		&& git submodule update --init vision/medical_imaging/3d-unet/nnUnet

# Clone power-dev repo.
.PHONY: clone_power_dev
clone_power_dev:
	@if [ ! -d $(POWER_DEV_DIR) ]; then \
		echo "Cloning Official Power-Dev repo" \
			&& git clone $(POWER_DEV_URL) $(POWER_DEV_DIR); \
	fi
	@echo "Updating Power-Dev repo" \
		&& cd $(POWER_DEV_DIR) \
		&& git fetch \
		&& git checkout $(POWER_DEV_HASH)

# Build Triton.
.PHONY: build_triton
build_triton:
ifeq ($(ARCH), x86_64)
	@echo "Building TensorRT Inference Server..."
	@if [ ! -d $(TRITON_DIR) ]; then \
		echo "triton-inference-server does not exist! Please exit the container and run make prebuild again." \
			&& exit 1; \
	fi
	@mkdir -p $(TRITON_OUT_DIR) \
		&& cd $(TRITON_OUT_DIR) \
		&& cmake -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
			-DTRITON_COMMON_REPO_TAG=mlperf-inference-v1.0 \
			-DTRITON_CORE_REPO_TAG=mlperf-inference-v1.0 \
			-DTRITON_BACKEND_REPO_TAG=mlperf-inference-v1.0 \
			-DTRITON_THIRD_PARTY_REPO_TAG=mlperf-inference-v1.0 \
			-DTRITON_ENABLE_TENSORRT=ON \
			-DTRITON_ENABLE_TENSORFLOW=OFF \
			-DTRITON_ENABLE_CAFFE2=OFF \
			-DTRITON_ENABLE_ONNXRUNTIME=OFF \
			-DTRITON_ENABLE_ONNXRUNTIME_OPENVINO=OFF \
			-DTRITON_ENABLE_PYTORCH=OFF \
			-DTRITON_ENABLE_CUSTOM=OFF \
			-DTRITON_ENABLE_HTTP=OFF \
			-DTRITON_ENABLE_GRPC=OFF \
			-DTRITON_ENABLE_METRICS=OFF \
			-DTRITON_ENABLE_METRICS_GPU=OFF \
			-DTRITON_ENABLE_GCS=OFF \
			-DTRITON_ENABLE_S3=OFF \
			-DTRITON_ENABLE_TRACING=OFF \
			-DTRITON_ENABLE_ASAN=OFF \
			-DTRITON_ENABLE_GPU=ON \
			-DTRITON_ENABLE_LOGGING=OFF \
			-DTRITON_ENABLE_STATS=OFF \
			-DTRITON_ENABLE_NVTX=OFF \
			../build \
		&& make -j server
endif

# Build TensorRT plugins.
.PHONY: build_plugins
build_plugins:
	mkdir -p build/plugins/NMSOptPlugin
	cd build/plugins/NMSOptPlugin \
		&& cmake -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) $(PROJECT_ROOT)/code/plugin/NMSOptPlugin \
		&& make -j
ifeq ($(ARCH), x86_64)
	mkdir -p build/plugins/DLRMInteractionsPlugin
	cd build/plugins/DLRMInteractionsPlugin \
		&& cmake -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) $(PROJECT_ROOT)/code/plugin/DLRMInteractionsPlugin \
		&& make -j
	mkdir -p build/plugins/DLRMBottomMLPPlugin
	cd build/plugins/DLRMBottomMLPPlugin \
		&& cmake -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) $(PROJECT_ROOT)/code/plugin/DLRMBottomMLPPlugin \
		&& make -j
endif
	mkdir -p build/plugins/RNNTOptPlugin
	cd build/plugins/RNNTOptPlugin \
		&& cmake -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) $(PROJECT_ROOT)/code/plugin/RNNTOptPlugin \
		&& make -j
	mkdir -p build/plugins/instanceNormalization3DPlugin
	cd build/plugins/instanceNormalization3DPlugin \
		&& cmake -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) $(PROJECT_ROOT)/code/plugin/instanceNormalization3DPlugin \
		&& make -j
	mkdir -p build/plugins/pixelShuffle3DPlugin
	cd build/plugins/pixelShuffle3DPlugin \
		&& cmake -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) $(PROJECT_ROOT)/code/plugin/pixelShuffle3DPlugin \
		&& make -j
	mkdir -p build/plugins/conv3D1X1X1K4Plugin
	cd build/plugins/conv3D1X1X1K4Plugin \
		&& cmake -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) $(PROJECT_ROOT)/code/plugin/conv3D1X1X1K4Plugin \
		&& make -j

# Build LoadGen.
.PHONY: build_loadgen
build_loadgen:
	@echo "Building loadgen..."
	@if [ ! -e $(LOADGEN_LIB_DIR) ]; then \
		mkdir $(LOADGEN_LIB_DIR); \
	fi
	@cd $(LOADGEN_LIB_DIR) \
		&& cmake -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) .. \
		&& make -j
	@if [ ! -e $(LOADGEN_LIB_DIR)/mlperf_loadgen.cpython-36m-$(ARCH)-linux-gnu.so ]; then \
		cd $(LOADGEN_INCLUDE_DIR) \
			&& CFLAGS="-std=c++14 -O3" python3 setup.py bdist_wheel \
			&& python3 -m pip install --force-reinstall -t $(LOADGEN_LIB_DIR) dist/mlperf_loadgen-*-linux_$(ARCH).whl; \
	fi

# Build harness source codes.
.PHONY: build_harness
build_harness:
	@echo "Building harness..."
	@mkdir -p build/harness \
		&& cd build/harness \
		&& cmake -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) -DLOADGEN_INCLUDE_DIR=$(LOADGEN_INCLUDE_DIR) -DLOADGEN_LIB_DIR=$(LOADGEN_LIB_DIR) $(PROJECT_ROOT)/code/harness \
		&& make -j
	@echo "Finished building harness."

# Compile scripts
build_scripts:
	@nvcc scripts/get_gencode.cu -o scripts/get_gencode_$(ARCH)

#######################  BUILD - TRITON CPU  #######################
# Build required TF and OV backends
.PHONY: build_triton_cpu_backends
build_triton_cpu_backends:
	echo "Building OV Triton backend libraries..."
	./scripts/build_triton_cpu_libs.sh

# Build all source codes for Triton-CPU.
.PHONY: build_cpu
build_cpu: clone_loadgen clone_power_dev clone_triton link_dirs
	@$(MAKE) -f $(MAKEFILE_NAME) build_triton_cpu
	@$(MAKE) -f $(MAKEFILE_NAME) build_loadgen
	@$(MAKE) -f $(MAKEFILE_NAME) build_harness_cpu

# Build Triton-CPU config
.PHONY: build_triton_cpu
build_triton_cpu:
ifeq ($(ARCH), x86_64)
	@echo "Building TensorRT Inference Server..."
	@if [ ! -d $(TRITON_DIR) ]; then \
		echo "triton-inference-server does not exist! Please exit the container and run make prebuild again." \
			&& exit 1; \
	fi
	@mkdir -p $(TRITON_OUT_DIR) \
		&& cd $(TRITON_OUT_DIR) \
		&& cmake -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
			-DTRITON_COMMON_REPO_TAG=mlperf-inference-v1.0 \
			-DTRITON_CORE_REPO_TAG=mlperf-inference-v1.0 \
			-DTRITON_BACKEND_REPO_TAG=mlperf-inference-v1.0 \
			-DTRITON_THIRD_PARTY_REPO_TAG=mlperf-inference-v1.0 \
			-DTRITON_ENABLE_TENSORRT=OFF \
			-DTRITON_ENABLE_TENSORFLOW=OFF \
			-DTRITON_ENABLE_CAFFE2=OFF \
			-DTRITON_ENABLE_ONNXRUNTIME=OFF \
			-DTRITON_ENABLE_ONNXRUNTIME_OPENVINO=OFF \
			-DTRITON_ENABLE_PYTORCH=OFF \
			-DTRITON_ENABLE_CUSTOM=OFF \
			-DTRITON_ENABLE_HTTP=OFF \
			-DTRITON_ENABLE_GRPC=OFF \
			-DTRITON_ENABLE_METRICS=OFF \
			-DTRITON_ENABLE_METRICS_GPU=OFF \
			-DTRITON_ENABLE_GCS=OFF \
			-DTRITON_ENABLE_S3=OFF \
			-DTRITON_ENABLE_TRACING=OFF \
			-DTRITON_ENABLE_ASAN=OFF \
			-DTRITON_ENABLE_GPU=ON \
			-DTRITON_ENABLE_LOGGING=ON \
			-DTRITON_ENABLE_STATS=ON \
			-DTRITON_ENABLE_NVTX=OFF \
			../build \
		&& make -j server
endif

# Build Triton-CPU harness source codes.
.PHONY: build_harness_cpu
build_harness_cpu:
	@echo "Building Triton-CPU harness..."
	@mkdir -p build/harness \
		&& cd build/harness \
		&& cmake -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) -DLOADGEN_INCLUDE_DIR=$(LOADGEN_INCLUDE_DIR) -DLOADGEN_LIB_DIR=$(LOADGEN_LIB_DIR) -DUSE_CPU=$(USE_CPU) $(PROJECT_ROOT)/code/harness \
		&& make -j
	@echo "Finished building Triton-CPU harness."

###############################  RUN  ###############################

# Generate TensorRT engines (plan files) and run the harness.
.PHONY: run
run:
	@$(MAKE) -f $(MAKEFILE_NAME) generate_engines
	@$(MAKE) -f $(MAKEFILE_NAME) run_harness

# Generate TensorRT engines (plan files).
.PHONY: generate_engines
generate_engines: link_dirs
	@python3 code/main.py $(RUN_ARGS) --action="generate_engines"

# Run the harness and check accuracy if in AccuracyOnly mode.
.PHONY: run_harness
run_harness: link_dirs
	@python3 code/main.py $(RUN_ARGS) --action="run_harness"
	@python3 scripts/print_harness_result.py $(RUN_ARGS)

# Run CPU-Triton harness. Update LD_LIBRARY_PATH with prebuild backend libraries
.PHONY: run_cpu_harness
run_cpu_harness: link_dirs
	LD_LIBRARY_PATH=/work/prebuilt_triton_libs:$(LD_LIBRARY_PATH) python3 code/main.py $(RUN_ARGS) --action="run_cpu_harness"
	LD_LIBRARY_PATH=/work/prebuilt_triton_libs:$(LD_LIBRARY_PATH) python3 scripts/print_harness_result.py $(RUN_ARGS)

# Run the harness and measure power consumption using official MLPerf-Inference power workflow.
.PHONY: run_harness_power
run_harness_power: link_dirs
	@$(MAKE) -f $(MAKEFILE_NAME) power_prologue
	python3.7 $(POWER_CLIENT_SCRIPT) -a $(POWER_SERVER_IP) -n "$(POWER_NTP_URL)" -s -S \
		-L $(POWER_LOGS_TEMP_DIR) -o $(POWER_LOGS_DIR) -f \
		-w 'LOG_DIR=$(POWER_LOGS_TEMP_DIR) python3 code/main.py $(RUN_ARGS) --action="run_harness" \
			&& cp -v $(POWER_LOGS_TEMP_DIR)/*/*/*/mlperf_log_detail.txt $(POWER_LOGS_TEMP_DIR)/ \
			&& cp -v $(POWER_LOGS_TEMP_DIR)/*/*/*/mlperf_log_summary.txt $(POWER_LOGS_TEMP_DIR)/'
	@$(MAKE) -f $(MAKEFILE_NAME) power_epilogue

.PHONY: run_audit_harness
run_audit_harness: link_dirs
	@$(MAKE) -f $(MAKEFILE_NAME) run_audit_test01
	@$(MAKE) -f $(MAKEFILE_NAME) run_audit_test04
	@$(MAKE) -f $(MAKEFILE_NAME) run_audit_test05

AUDIT_HARNESS := run_audit_harness
ifeq ($(USE_CPU), 1)
	AUDIT_HARNESS := run_cpu_audit_harness
endif

AUDIT_VERIFICATION := run_audit_verification
ifeq ($(USE_CPU), 1)
	AUDIT_VERIFICATION := run_cpu_audit_verification
endif

.PHONY: run_audit_test01
run_audit_test01: link_dirs
	@python3 code/main.py $(RUN_ARGS) --audit_test=TEST01 --action="$(AUDIT_HARNESS)"
	@python3 code/main.py $(RUN_ARGS) --audit_test=TEST01 --action="$(AUDIT_VERIFICATION)"

.PHONY: run_audit_test04_once
run_audit_test04_once: link_dirs
	@echo "Sleep to reset thermal state before TEST04-A..." && sleep 20
	@python3 code/main.py $(RUN_ARGS) --audit_test=TEST04-A --action="$(AUDIT_HARNESS)"
	@echo "Sleep to reset thermal state before TEST04-B..." && sleep 20
	@python3 code/main.py $(RUN_ARGS) --audit_test=TEST04-B --action="$(AUDIT_HARNESS)"
	@python3 code/main.py $(RUN_ARGS) --audit_test=TEST04-A --action="$(AUDIT_VERIFICATION)"

# TEST04 is so short that it is sometimes unstable. Try up to three times before failing.
.PHONY: run_audit_test04
run_audit_test04:
	@for i in 1 2 3; do echo "TEST04 trial $$i" && $(MAKE) -f $(MAKEFILE_NAME) run_audit_test04_once && break; done

.PHONY: run_audit_test05_once
run_audit_test05_once: link_dirs
	@echo "Sleep to reset thermal state before TEST05..." && sleep 20
	@python3 code/main.py $(RUN_ARGS) --audit_test=TEST05 --action="$(AUDIT_HARNESS)"
	@python3 code/main.py $(RUN_ARGS) --audit_test=TEST05 --action="$(AUDIT_VERIFICATION)"

# TEST05 is sometimes unstable. Try up to two times before failing.
.PHONY: run_audit_test05
run_audit_test05:
	@for i in 1 2; do echo "TEST05 trial $$i" && $(MAKE) -f $(MAKEFILE_NAME) run_audit_test05_once && break; done

.PHONY: lock_clocks
lock_clocks:
ifeq ($(ARCH), x86_64)
	@echo "Locking clocks to $(GPUCLK)"
	@sudo nvidia-smi -lgc $(GPUCLK),$(GPUCLK)
else
	@echo "Locking Xavier clocks to max freq (on cl-63 it is 1.377 GHz). "
	@sudo jetson_clocks
endif

.PHONY: free_clocks
free_clocks:
ifeq ($(ARCH), x86_64)
	@echo "Freeing clocks"
	@sudo nvidia-smi -rgc
else
	@echo "Freeing clocks on Xavier is not implemented yet. "
endif

.PHONY: copy_profiles
copy_profiles:
	@echo "Copying yml and sqlite files"
	@cp *.yml /home/scratch.dlsim/data/mlperf-inference/
	@cp *.sqlite /home/scratch.dlsim/data/mlperf-inference/

.PHONY: run_harness_correlation
run_harness_correlation: link_dirs
	@$(MAKE) -f $(MAKEFILE_NAME) lock_clocks
	@$(MAKE) -f $(MAKEFILE_NAME) run_harness || ($(MAKE) -f $(MAKEFILE_NAME) free_clocks ; exit 1)
	@$(MAKE) -f $(MAKEFILE_NAME) free_clocks

# Re-generate TensorRT calibration cache.
.PHONY: calibrate
calibrate: link_dirs
	@python3 code/main.py $(RUN_ARGS) --action="calibrate"

############################## POWER SUBMISSION ##############################
.PHONY: power_prologue
power_prologue:
	@mkdir -p $(POWER_LOGS_TEMP_DIR)
	@mkdir -p $(POWER_LOGS_DIR)
	@echo "Checking if someone else is using the Power Server machine..."
	@$(MAKE) -f $(MAKEFILE_NAME) power_check_server_not_running || $(MAKE) -f $(MAKEFILE_NAME) power_kill_server
	@echo "Copying Server config to the Power Server..."
	$(POWER_SCP) $(POWER_SERVER_CONFIG) $(POWER_SERVER_USER_IP):$(POWER_SERVER_SCRIPT_DIR)/server.cfg
	@echo "Running Server script on the Power Server..."
	$(POWER_SSH) $(POWER_SERVER_USER_IP) \
		powershell -Command "Start-Process python -PassThru -Wait \
		-ArgumentList '$(POWER_SERVER_SCRIPT_DIR)/server.py -c $(POWER_SERVER_SCRIPT_DIR)/server.cfg' \
		-WorkingDirectory $(POWER_SERVER_SCRIPT_DIR) -WindowStyle Hidden \
		-RedirectStandardOutput $(POWER_SERVER_SCRIPT_DIR)/server_stdout.log \
		-RedirectStandardError $(POWER_SERVER_SCRIPT_DIR)/server.log" &
	@echo "Checking if Power Server script was successfully started..."
	@for i in {1..3}; do $(MAKE) -f $(MAKEFILE_NAME) power_check_server_running && break; done
	@echo "Power Server is running and listening to client now."

.PHONY: power_check_server_running
power_check_server_running:
	@echo "Checking if Power Server is currently running..."
	@$(POWER_SSH) $(POWER_SERVER_USER_IP) \
		powershell -Command "Get-Process python -ErrorAction SilentlyContinue"

.PHONY: power_check_server_not_running
power_check_server_not_running:
	@echo "Checking if Power Server is NOT currently running..."
	@! $(POWER_SSH) $(POWER_SERVER_USER_IP) \
		powershell -Command "Get-Process python -ErrorAction SilentlyContinue"

.PHONY: power_kill_server
power_kill_server:
	@echo "Killing server.py on Power Server..."
	@$(POWER_SSH) $(POWER_SERVER_USER_IP) \
		powershell -Command "taskkill /IM python.exe /F"

.PHONY: power_epilogue
power_epilogue:
	@echo "Detecting session name..."
	@$(eval SESSION_NAME=$(shell bash -c "ls -td $(POWER_LOGS_DIR)/*/ | head -n 1 | rev | cut -d / -f 2 | rev"))
	@echo "Session name: $(SESSION_NAME)"
	@echo "Downloading Server logs..."
	$(POWER_SCP) $(POWER_SERVER_USER_IP):$(POWER_SERVER_LOG_DIR)/$(SESSION_NAME)/ranging/spl.txt \
		$(POWER_LOGS_DIR)/$(SESSION_NAME)/ranging/spl.txt
	$(POWER_SCP) $(POWER_SERVER_USER_IP):$(POWER_SERVER_LOG_DIR)/$(SESSION_NAME)/run_1/spl.txt \
		$(POWER_LOGS_DIR)/$(SESSION_NAME)/run_1/spl.txt
	$(POWER_SCP) $(POWER_SERVER_USER_IP):$(POWER_SERVER_LOG_DIR)/$(SESSION_NAME)/power/ptd_logs.txt \
		$(POWER_LOGS_DIR)/$(SESSION_NAME)/power/ptd_logs.txt
	$(POWER_SCP) $(POWER_SERVER_USER_IP):$(POWER_SERVER_LOG_DIR)/$(SESSION_NAME)/power/server.json \
		$(POWER_LOGS_DIR)/$(SESSION_NAME)/power/server.json
	$(POWER_SCP) $(POWER_SERVER_USER_IP):$(POWER_SERVER_LOG_DIR)/$(SESSION_NAME)/power/server.log \
		$(POWER_LOGS_DIR)/$(SESSION_NAME)/power/server.log
	@rm $(POWER_LOGS_DIR)/$(SESSION_NAME)/*/mlperf_log*
	@LOG_DIR=$(POWER_LOGS_DIR)/$(SESSION_NAME)/run_1 python3 scripts/print_harness_result.py $(RUN_ARGS)
	@echo "Power logs are located in $(POWER_LOGS_DIR)/$(SESSION_NAME)."

.PHONY: power_set_maxq_state
power_set_maxq_state:
ifeq ($(HOSTNAME_SHORT), computelab-ro-prod-01)
	@echo "Setting MaxQ power state on NVIDIA DGX Station A100 machine"
	$(SHELL) power/set_nvlink.sh 0
	sudo nvidia-smi -pl 250
else ifeq ($(HOSTNAME_SHORT), ipp1-1469)
	@echo "Setting MaxQ power state on A100-PCIex8 machine"
	source power/set_fan_speed.sh && MACHINE_TYPE=GIGABYTE set_fan_auto
	$(SHELL) power/set_nvlink.sh 0
	sudo nvidia-smi -pl 200
else ifeq ($(HOSTNAME_SHORT), sjc1-luna-02)
	@echo "Setting MaxQ power state on NVIDIA DGX A100 machine"
	source power/set_fan_speed.sh && MACHINE_TYPE=LUNA set_fan_auto && sleep 10
	$(SHELL) power/check_switch.sh && sleep 10
	$(SHELL) power/set_nvswitch.sh 0 && sleep 10
	$(SHELL) power/set_nvlink.sh 0 && sleep 10
	sudo nvidia-smi -pm 1
	sudo nvidia-smi -pl 275
else ifeq ($(HOSTNAME_SHORT), computelab-310)
	@echo "Setting MaxQ power state on NVIDIA AGX Xavier"
	sudo /usr/sbin/nvpmodel -f power/nvpmodel-computelab-310.conf -m 8
	sudo /usr/sbin/nvpmodel -d cool
	#sudo bash -c 'echo "14100000.pcie" > /sys/bus/platform/drivers/tegra-pcie-dw/unbind'
else ifeq ($(HOSTNAME_SHORT), computelab-501)
	@echo "Setting MaxQ power state on NVIDIA Xavier NX"
	sudo /usr/sbin/nvpmodel -f power/nvpmodel-computelab-501.conf -m 6
	sudo /usr/sbin/nvpmodel -d cool
	#sudo bash -c "echo 5000 > /sys/devices/c250000.i2c/i2c-7/7-0040/iio:device0/crit_current_limit_0"
	#sudo bash -c "echo 5000 > /sys/devices/c250000.i2c/i2c-7/7-0040/iio:device0/warn_current_limit_0"
else
	@echo "MaxQ state is not supported on this machine!"
	@exit 1
endif

.PHONY: power_reset_maxq_state
power_reset_maxq_state:
ifeq ($(HOSTNAME_SHORT), computelab-ro-prod-01)
	@echo "Resetting power state on NVIDIA DGX Station A100 machine"
	$(SHELL) power/set_nvlink.sh 1
	sudo nvidia-smi -pl 275
else ifeq ($(HOSTNAME_SHORT), ipp1-1469)
	@echo "Resetting power state on A100-PCIex8 machine"
	$(SHELL) power/set_nvlink.sh 1
	source power/set_fan_speed.sh && MACHINE_TYPE=GIGABYTE set_fan_manual_speed 0xFF
	sudo nvidia-smi -pl 250
else ifeq ($(HOSTNAME_SHORT), sjc1-luna-02)
	@echo "Resetting power state on NVIDIA DGX A100 machine"
	source power/set_fan_speed.sh && MACHINE_TYPE=LUNA set_fan_manual_speed 100 && sleep 10
	$(SHELL) power/set_nvlink.sh 1 && sleep 10
	$(SHELL) power/set_nvswitch.sh 1 && sleep 10
	sudo nvidia-smi -pm 1
	sudo nvidia-smi -pl 400
else ifeq ($(HOSTNAME_SHORT), computelab-310)
	@echo "Resetting power state on NVIDIA AGX Xavier"
	@sudo /usr/sbin/nvpmodel -f /etc/nvpmodel.conf -m 0
	@sudo /usr/sbin/nvpmodel -d cool
else ifeq ($(HOSTNAME_SHORT), computelab-501)
	@echo "Resetting power state on NVIDIA Xavier NX"
	@sudo /usr/sbin/nvpmodel -f /etc/nvpmodel.conf -m 0
	@sudo /usr/sbin/nvpmodel -d cool
	#sudo bash -c "echo 5000 > /sys/devices/c250000.i2c/i2c-7/7-0040/iio:device0/crit_current_limit_0"
	#sudo bash -c "echo 5000 > /sys/devices/c250000.i2c/i2c-7/7-0040/iio:device0/warn_current_limit_0"
else
	@echo "$(HOSTNAME_SHORT) does not have MaxQ submission."
endif

############################## AUTOMATION AND SUBMISSION ##############################
# RUN_ID: ID for the run. For L1, this is the number of the run
# SYSTEM_NAME: Name of the current platform
# ARTIFACT_SRC_PATH: path/to/directory to compress
# ARTIFACT_NAME: name-of-artifact-to-push
# ARTIFACT_DST_PATH: path/in/artifactory/to/prepend
ARTIFACTORY_URL := https://urm.nvidia.com/artifactory/sw-mlpinf-generic

RUN_ID ?= manual-$(TIMESTAMP)
SYSTEM_NAME ?= $(shell python3 scripts/get_system_id.py 2> /dev/null)
ARTIFACT_NAME ?= $(SYSTEM_NAME)_$(RUN_ID)
ARTIFACT_DST_PATH ?= artifacts

ARTIFACT_URL := $(ARTIFACTORY_URL)/$(ARTIFACT_DST_PATH)/$(ARTIFACT_NAME).gz

# Generate a raw results directory in build/full_results from LoadGen logs in build/logs
.PHONY: update_results
update_results:
	@python3 scripts/update_results.py --output_dir results --result_id $(ARTIFACT_NAME)
	@printf "If you would like to push results, run:\n\tmake truncate_results\n\tmake push_full_results ARTIFACT_NAME=$(ARTIFACT_NAME)\n"

.PHONY: update_compliance
update_compliance:
	@python3 scripts/update_results.py --input_dir build/compliance_logs --output_dir compliance --assume_compliance --result_id $(ARTIFACT_NAME)
	@printf "If you would like to push results, run:\n\tmake truncate_results\n\tmake push_full_results ARTIFACT_NAME=$(ARTIFACT_NAME)\n"

.PHONY: truncate_results
truncate_results:
	@echo "WARNING: This script cannot be executed from within the docker container."
	@echo "It must have access to the project root at ../../"
	@rm -rf build/full_results
	@cd ../../ \
		&& python3 closed/$(SUBMITTER)/build/inference/tools/submission/truncate_accuracy_log.py --input . --backup closed/$(SUBMITTER)/build/full_results --submitter $(SUBMITTER)
	@echo "Full accuracy logs stored in build/full_results/. Truncated results stored in results/."

.PHONY: summarize_results
summarize_results:
	@python3 scripts/internal/results_analysis/summarize_results.py

.PHONY: check_submission
check_submission:
	@echo "WARNING: This script cannot be executed from within the docker container."
	@echo "It must have access to the project root at ../../"
	@cd ../../ \
		&& python3 closed/$(SUBMITTER)/build/inference/tools/submission/submission-checker.py --input . --submitter $(SUBMITTER) 2>&1 \
		| tee closed/$(SUBMITTER)/results/submission_checker_log.txt

.PHONY: check_submission_power
check_submission_power:
	@cd ../../ \
		&& python3.7 closed/$(SUBMITTER)/build/inference/tools/submission/submission-checker.py --more-power-check --input . --submitter $(SUBMITTER) 2>&1 \
		| tee closed/$(SUBMITTER)/results/submission_checker_log.txt

.PHONY: push_artifacts
push_artifacts:
	@mkdir -p build/artifacts
	@tar -cvzf build/artifacts/$(ARTIFACT_NAME).gz $(ARTIFACT_SRC_PATH)
	curl -u$(UNAME):$(ARTIFACTORY_API_KEY) -T build/artifacts/$(ARTIFACT_NAME).gz "$(ARTIFACT_URL)"

.PHONY: push_full_results
push_full_results:
	@$(MAKE) -f $(MAKEFILE_NAME) push_artifacts ARTIFACT_SRC_PATH=build/full_results ARTIFACT_DST_PATH=full_result_logs ARTIFACT_NAME=full-results_$(ARTIFACT_NAME)

############################## UTILITY ##############################

.PHONY: generate_conf_files
generate_conf_files:
	@python3 scripts/create_config_files.py

# Remove build directory.
.PHONY: clean
clean: clean_shallow
	rm -rf build

# Remove only the files necessary for a clean build.
.PHONY: clean_shallow
clean_shallow:
	rm -rf build/bin
	rm -rf build/harness
	rm -rf build/plugins
	rm -rf $(TRITON_OUT_DIR)
	rm -rf $(LOADGEN_LIB_DIR)

# Print out useful information.
.PHONY: info
info:
	@echo "RUN_ID=$(RUN_ID)"
	@echo "SYSTEM_NAME=$(SYSTEM_NAME)"
	@echo "Architecture=$(ARCH)"
	@echo "SM_GENCODE=$(SM_GENCODE)"
	@echo "User=$(UNAME)"
	@echo "UID=$(UID)"
	@echo "HOSTNAME=$(HOSTNAME)"
	@echo "Usergroup=$(GROUPNAME)"
	@echo "GroupID=$(GROUPID)"
	@echo "Docker info: {DETACH=$(DOCKER_DETACH), TAG=$(DOCKER_TAG)}"
ifdef DOCKER_IMAGE_NAME
	@echo "Docker image used: $(DOCKER_IMAGE_NAME) -> [$(BASE_IMAGE)]"
endif
	@echo "PATH=$(PATH)"
	@echo "CPATH=$(CPATH)"
	@echo "CUDA_PATH=$(CUDA_PATH)"
	@echo "LIBRARY_PATH=$(LIBRARY_PATH)"
	@echo "LD_LIBRARY_PATH=$(LD_LIBRARY_PATH)"
	@echo "MIG_CONF=$(MIG_CONF)"

# The shell target will start a shell that inherits all the environment
# variables set by this Makefile for convenience.
.PHONY: shell
shell:
	@$(SHELL)
