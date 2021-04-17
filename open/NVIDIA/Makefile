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
GROUPNAME := $(shell id -gn `whoami`)
GROUPID := $(shell id -g `whoami`)

# Conditional Docker flags
ifndef DOCKER_DETACH
DOCKER_DETACH := 0
endif
ifndef DOCKER_TAG
DOCKER_TAG := $(UNAME)
endif

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
LOADGEN_INCLUDE_DIR := $(INFERENCE_DIR)/loadgen
LOADGEN_LIB_DIR := $(LOADGEN_INCLUDE_DIR)/build
INFERENCE_HASH = 6958607d52f646d61fb950341f523597482b10e3

# Set Environment variables to extracted contents
export LD_LIBRARY_PATH := /usr/local/cuda/lib64:/usr/lib/$(ARCH)-linux-gnu:$(LOADGEN_LIB_DIR):$(LD_LIBRARY_PATH)
export LIBRARY_PATH := /usr/local/cuda/lib64:/usr/lib/$(ARCH)-linux-gnu:$(LOADGEN_LIB_DIR):$(LIBRARY_PATH)
export PATH := /usr/local/cuda/bin:$(PATH)
export CPATH := /usr/local/cuda/include:/usr/include/$(ARCH)-linux-gnu:/usr/include/$(ARCH)-linux-gnu/cub:$(CPATH)
export CUDA_PATH := /usr/local/cuda
export CCACHE_DISABLE=1

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

# Specify default dir for harness output logs.
ifndef LOG_DIR
	export LOG_DIR := $(BUILD_DIR)/logs/$(shell date +'%Y.%m.%d-%H.%M.%S')
endif

# Specify debug options for build (default to Release build)
ifeq ($(DEBUG),1)
BUILD_TYPE := Debug
else
BUILD_TYPE := Release
endif

# Handle different nvidia-docker version
ifneq ($(wildcard /usr/bin/nvidia-docker),)
	DOCKER_RUN_CMD := nvidia-docker run
	# Set Environment variables to fix docker client and server version mismatch
	# Related issue: https://github.com/kubernetes-sigs/kubespray/issues/6160
	export DOCKER_API_VERSION=1.39
else
	DOCKER_RUN_CMD := docker run --gpus=all
endif

ifneq ($(ARCH), aarch64)
DOCKER_IMAGE_NAME := base

# Specify default base image
ifndef BASE_IMAGE
ifeq ($(shell bash $(PROJECT_ROOT)/scripts/check_intranet.sh),0)
	BASE_IMAGE := gitlab-master.nvidia.com/compute/mlperf-inference:$(DOCKER_IMAGE_NAME)-cuda11.0
else
	BASE_IMAGE := nvidia/cuda:11.0-cudnn8-devel-ubuntu18.04
endif # check_intranet
endif # ifndef BASE_IMAGE
endif # aarch64 check

############################## PREBUILD ##############################
INT4_DIR := $(PROJECT_ROOT)/code/resnet50/int4

ifeq ($(OPEN_BENCHMARK), resnet-int4)
	include $(INT4_DIR)/Makefile
endif

.PHONY: prebuild_int4
prebuild_int4:
	@$(MAKE) -f $(MAKEFILE_NAME) build_docker_int4 NO_BUILD?=1 BASE_IMAGE=$(BASE_IMAGE)
ifneq ($(strip ${DOCKER_DETACH}), 1)
	@$(MAKE) -f $(MAKEFILE_NAME) attach_docker OPEN_BENCHMARK=resnet-int4
endif

# Add symbolic links to scratch path if it exists. RESNET
.PHONY: link_dataset_dir
link_dataset_dir:
	@mkdir -p build
ifneq ($(MLPERF_INFERENCE_PATH),)
	@if [ ! -e $(DATA_DIR) ]; then \
		ln -sn $(MLPERF_INFERENCE_PATH)/data $(DATA_DIR); \
	fi
	@if [ ! -e $(PREPROCESSED_DATA_DIR) ]; then \
		ln -sn $(MLPERF_INFERENCE_PATH)/preprocessed_data $(PREPROCESSED_DATA_DIR); \
	fi
	@if [ ! -e $(MODEL_DIR) ]; then \
		ln -sn $(MLPERF_INFERENCE_PATH)/models $(MODEL_DIR); \
	fi
endif

# Add symbolic links to scratch path if it exists.
.PHONY: link_dirs
link_dirs:
	@mkdir -p build
ifdef MLPERF_SCRATCH_PATH
	@mkdir -p $(MLPERF_SCRATCH_PATH)/data
	@mkdir -p $(MLPERF_SCRATCH_PATH)/preprocessed_data
	@mkdir -p $(MLPERF_SCRATCH_PATH)/models
	@ln -sfn $(MLPERF_SCRATCH_PATH)/data $(DATA_DIR)
	@ln -sfn $(MLPERF_SCRATCH_PATH)/preprocessed_data $(PREPROCESSED_DATA_DIR)
	@ln -sfn $(MLPERF_SCRATCH_PATH)/models $(MODEL_DIR)
endif

# Build the docker image for resnet int4
.PHONY: build_docker_int4
build_docker_int4: 
ifeq ($(ARCH), x86_64)
	@echo "Building Docker image for ResNet INT4"
	DOCKER_BUILDKIT=1 docker build -t mlperf-inference:$(DOCKER_TAG)-latest \
		--build-arg BASE_IMAGE=$(BASE_IMAGE) \
		--network host - < docker/Dockerfile.int4
endif

# Add current user into docker image.
.PHONY: docker_add_user
docker_add_user:
ifeq ($(ARCH), x86_64)
	@echo "Adding user account into image"
	docker build -t mlperf-inference:$(DOCKER_TAG) --network host \
		--build-arg BASE_IMAGE=mlperf-inference:$(DOCKER_TAG)-latest \
		--build-arg GID=$(GROUPID) --build-arg UID=$(UID) --build-arg GROUP=$(GROUPNAME) --build-arg USER=$(UNAME) \
		- < docker/Dockerfile.user
endif

# Launch an interactive container
.PHONY: attach_docker
attach_docker: 
	@$(MAKE) -f $(MAKEFILE_NAME) docker_add_user
	@$(MAKE) -f $(MAKEFILE_NAME) launch_docker

.PHONY: launch_docker
launch_docker: 
ifeq ($(ARCH), x86_64)
	@echo "Launching Docker interactive session"
	$(DOCKER_RUN_CMD) -e OPEN_BENCHMARK=$(OPEN_BENCHMARK) --rm -ti -w /work -v $(HOST_VOL):$(CONTAINER_VOL) -v ${HOME}:/mnt/${HOME} \
		-v /etc/timezone:/etc/timezone:ro -v /etc/localtime:/etc/localtime:ro \
		$(DOCKER_MOUNTS) \
		--security-opt apparmor=unconfined --security-opt seccomp=unconfined \
		--name $(DOCKER_NAME) -h $(DOCKER_NAME) --add-host mlperf-inference-$(UNAME):127.0.0.1 \
		`if [ -d /home/scratch.mlperf_inference ]; then echo "-v /home/scratch.mlperf_inference:/home/scratch.mlperf_inference"; fi` \
		`if [ -d /scratch/datasets/mlperf_inference ]; then echo "-v /scratch/datasets/mlperf_inference:/scratch/datasets/mlperf_inference"; fi` \
		`if [ -d /home/scratch.dlsim ]; then echo "-v /home/scratch.dlsim:/home/scratch.dlsim"; fi` \
		`if [ -d /gpfs/fs1/datasets/mlperf_inference ]; then echo "-v /gpfs/fs1/datasets/mlperf_inference:/gpfs/fs1/datasets/mlperf_inference"; fi` \
		-v $(PROJECT_ROOT)/../../closed/NVIDIA/code/:/closed \
		--user $(UID):$(GROUPID) --net host --device /dev/fuse --cap-add SYS_ADMIN $(DOCKER_ARGS) mlperf-inference:$(DOCKER_TAG)
endif	

############################## PREPROCESS_DATA ##############################

.PHONY: preprocess_data
preprocess_data: link_dirs
	@python3 scripts/preprocess_data.py -d $(DATA_DIR) -o build/preprocessed_data

############################### BUILD ###############################
.PHONY: build
build: clone_loadgen link_dirs
ifeq ($(ARCH), x86_64)
	@$(MAKE) -f $(MAKEFILE_NAME) build_loadgen
ifeq ($(OPEN_BENCHMARK), resnet-int4)
	@$(MAKE) -f $(MAKEFILE_NAME) build_int4
endif
endif

# Clone LoadGen repo.
.PHONY: clone_loadgen
clone_loadgen:
	@if [ ! -d $(LOADGEN_INCLUDE_DIR) ]; then \
		echo "Cloning Official MLPerf Inference (For Loadgen Files)" \
			&& git clone https://github.com/mlperf/inference.git $(INFERENCE_DIR); \
	fi
	@echo "Updating Loadgen" \
		&& cd $(INFERENCE_DIR) \
		&& git fetch \
		&& git checkout $(INFERENCE_HASH) \
		&& git submodule update --init third_party/pybind

# Build loadGen
.PHONY: build_loadgen
build_loadgen:
	@echo "Building loadgen..."
ifeq ($(ARCH), x86_64)
	@if [ ! -e $(LOADGEN_LIB_DIR) ]; then \
		mkdir $(LOADGEN_LIB_DIR); \
	fi
	@cd $(LOADGEN_LIB_DIR) \
		&& cmake -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) .. \
		&& make -j
endif

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

.PHONY: run_audit_harness
run_audit_harness: link_dirs
	@$(MAKE) -f $(MAKEFILE_NAME) run_audit_test01
	@$(MAKE) -f $(MAKEFILE_NAME) run_audit_test04
	@$(MAKE) -f $(MAKEFILE_NAME) run_audit_test05

.PHONY: run_audit_test01
run_audit_test01: link_dirs
	@python3 code/main.py $(RUN_ARGS) --audit_test=TEST01 --action="run_audit_harness"
	@python3 code/main.py $(RUN_ARGS) --audit_test=TEST01 --action="run_audit_verification"

.PHONY: run_audit_test04
run_audit_test04: link_dirs
	@python3 code/main.py $(RUN_ARGS) --audit_test=TEST04-A --action="run_audit_harness"
	@python3 code/main.py $(RUN_ARGS) --audit_test=TEST04-B --action="run_audit_harness"
	@python3 code/main.py $(RUN_ARGS) --audit_test=TEST04-A --action="run_audit_verification"

.PHONY: run_audit_test05
run_audit_test05: link_dirs
	@python3 code/main.py $(RUN_ARGS) --audit_test=TEST05 --action="run_audit_harness"
	@python3 code/main.py $(RUN_ARGS) --audit_test=TEST05 --action="run_audit_verification"

############################## UTILITY ##############################
.PHONY: clean
clean: clean_shallow
	rm -rf build

# For clean build.
.PHONY: clean_shallow
clean_shallow:
	rm -rf build/bin
	rm -rf build/harness
	rm -rf $(LOADGEN_LIB_DIR)
ifeq ($(ARCH), x86_64)
	@echo "Cleaning INT4 harness..."
	cd $(INT4_DIR) \
		&& make -j CUDA=$(CUDA_PATH) BUILD_DIR=$(INT4_DIR) LOADGEN_PATH=$(LOADGEN_INCLUDE_DIR) clean
endif

.PHONY: info
info:
	@echo "Architecture=$(ARCH)"
	@echo "User=$(UNAME)"
	@echo "UID=$(UID)"
	@echo "Usergroup=$(GROUPNAME)"
	@echo "GroupID=$(GROUPID)"
	@echo "Docker info: {DETACH=$(DOCKER_DETACH), TAG=$(DOCKER_TAG)}"
ifdef DOCKER_IMAGE_NAME
	@echo "Docker image used: $(DOCKER_IMAGE_NAME) -> [$(BASE_IMAGE)]"
endif
	@echo "CUDA Version=$(CUDA_VER)"
ifdef DRIVER_VER
	@echo "NVidia Driver Version=$(DRIVER_VER)"
endif
	@echo "PATH=$(PATH)"
	@echo "CPATH=$(CPATH)"
	@echo "CUDA_PATH=$(CUDA_PATH)"
	@echo "LIBRARY_PATH=$(LIBRARY_PATH)"
	@echo "LD_LIBRARY_PATH=$(LD_LIBRARY_PATH)"

# The shell target will start a shell that inherits all the environment
# variables set by this Makefile for convenience.
.PHONY: shell
shell:
	@$(SHELL)
