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

SUBMITTER ?= Alibaba
DIVISION ?= open

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
LOADGEN_INCLUDE_DIR := $(INFERENCE_DIR)/loadgen
LOADGEN_LIB_DIR := $(LOADGEN_INCLUDE_DIR)/build
INFERENCE_HASH = 6958607d52f646d61fb950341f523597482b10e3

# Set the Triton directory
TRITON_DIR = $(BUILD_DIR)/triton-inference-server
TRITON_OUT_DIR = $(TRITON_DIR)/out
TRITON_URL = https://github.com/triton-inference-server/server
TRITON_HASH = d3712aad9db49ac054d950e957a955eff77e6903

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
    export LOG_DIR := $(BUILD_DIR)/logs/$(TIMESTAMP)
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
    export DOCKER_API_VERSION=1.40
else
    DOCKER_RUN_CMD := docker run --gpus=all
endif

# If DOCKER_COMMAND is not pass, launch interactive docker container session.
ifneq ($(DOCKER_COMMAND),"")
	DOCKER_INTERACTIVE_FLAGS := -it
else
	DOCKER_INTERACTIVE_FLAGS :=
endif

ALLOWED_MIG_CONFS := OFF 1 2 3 ALL
MIG_CONF ?= OFF
ifeq ($(filter $(MIG_CONF),$(ALLOWED_MIG_CONFS)),)
    $(warning MIG_CONF was not set to a valid value. Default to OFF to be safe.)
    MIG_CONF := OFF
endif

ifneq ($(ARCH), aarch64)
    DRIVER_VER_MAJOR ?= $(shell nvidia-smi | /bin/grep -Eo 'Driver Version: [+-]?[0-9]+' | awk -F ' ' '{print $$NF}')

    # Check driver version and launch the appropriate container. If driver >= 455, use CUDA 11.1, otherwise 11.0
    ifeq ($(shell if [ $(DRIVER_VER_MAJOR) -ge 455 ]; then echo true; else echo false; fi), true)
        CUDA_VER := 11.1

        # Workaround: Some machines were upgraded to driver 460, but TRT build is listed under r455, so it
        # will fail to find the URL
        DRIVER_VER_MAJOR := 455
    else
        CUDA_VER := 11.0

        # Workaround: Our CUDA 11.0 submissions were upgraded to driver 450, but TRT build is listed under r445, so it
        # will fail to find the URL
        DRIVER_VER_MAJOR := 445
    endif # Driver check
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
.PHONY: prebuild
prebuild:
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
ifdef MLPERF_SCRATCH_PATH
	@mkdir -p $(MLPERF_SCRATCH_PATH)/data
	@mkdir -p $(MLPERF_SCRATCH_PATH)/preprocessed_data
	@mkdir -p $(MLPERF_SCRATCH_PATH)/models
	@ln -sfn $(MLPERF_SCRATCH_PATH)/data $(DATA_DIR)
	@ln -sfn $(MLPERF_SCRATCH_PATH)/preprocessed_data $(PREPROCESSED_DATA_DIR)
	@ln -sfn $(MLPERF_SCRATCH_PATH)/models $(MODEL_DIR)
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
	docker pull $(BASE_IMAGE)
endif
	DOCKER_BUILDKIT=$(DOCKER_BUILDKIT) docker build -t mlperf-inference:$(DOCKER_TAG)-latest \
		--build-arg BASE_IMAGE=$(BASE_IMAGE) \
		--build-arg CUDA_VER=$(CUDA_VER) \
		--build-arg DRIVER_VER_MAJOR=$(DRIVER_VER_MAJOR) \
		--build-arg SM_GENCODE=$(SM_GENCODE) \
		--network host -f docker/Dockerfile .
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
		--cap-add SYS_ADMIN \
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

############################### BUILD ###############################

# Build all source codes.
.PHONY: build
build: clone_loadgen link_dirs
#	@$(MAKE) -f $(MAKEFILE_NAME) build_triton
#	@$(MAKE) -f $(MAKEFILE_NAME) build_plugins
	@$(MAKE) -f $(MAKEFILE_NAME) build_loadgen
	@$(MAKE) -f $(MAKEFILE_NAME) build_harness

# Clone LoadGen repo.
.PHONY: clone_loadgen
clone_loadgen:
	@if [ ! -d $(LOADGEN_INCLUDE_DIR) ]; then \
		echo "Cloning Official MLPerf Inference (For Loadgen Files)" \
			&& git clone https://github.com/mlcommons/inference.git $(INFERENCE_DIR); \
	fi
	@echo "Updating Loadgen" \
		&& cd $(INFERENCE_DIR) \
		&& git fetch \
		&& git checkout $(INFERENCE_HASH) \
		&& git submodule update --init third_party/pybind \
		&& git submodule update --init language/bert/DeepLearningExamples \
		&& git submodule update --init vision/medical_imaging/3d-unet/nnUnet

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

.PHONY: run_audit_test04_once
run_audit_test04_once: link_dirs
	@echo "Sleep to reset thermal state before TEST04-A..." && sleep 20
	@python3 code/main.py $(RUN_ARGS) --audit_test=TEST04-A --action="run_audit_harness"
	@echo "Sleep to reset thermal state before TEST04-B..." && sleep 20
	@python3 code/main.py $(RUN_ARGS) --audit_test=TEST04-B --action="run_audit_harness"
	@python3 code/main.py $(RUN_ARGS) --audit_test=TEST04-A --action="run_audit_verification"

# TEST04 is so short that it is sometimes unstable. Try up to three times before failing.
.PHONY: run_audit_test04
run_audit_test04:
	@for i in 1 2 3; do echo "TEST04 trial $$i" && $(MAKE) -f $(MAKEFILE_NAME) run_audit_test04_once && break; done

.PHONY: run_audit_test05_once
run_audit_test05_once: link_dirs
	@echo "Sleep to reset thermal state before TEST05..." && sleep 20
	@python3 code/main.py $(RUN_ARGS) --audit_test=TEST05 --action="run_audit_harness"
	@python3 code/main.py $(RUN_ARGS) --audit_test=TEST05 --action="run_audit_verification"

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
		&& python3 ${DIVISION}/$(SUBMITTER)/build/inference/tools/submission/truncate_accuracy_log.py --input . --backup ${DIVISION}/$(SUBMITTER)/build/full_results --submitter $(SUBMITTER)
	@$(MAKE) -f $(MAKEFILE_NAME) summarize_results
	@echo "Full accuracy logs stored in build/full_results/. Truncated results stored in results/."

.PHONY: summarize_results
summarize_results:
	@python3 scripts/summarize_results.py \
		&& echo "Summary of perf results exported to results_summary.csv"

.PHONY: check_submission
check_submission:
	@echo "WARNING: This script cannot be executed from within the docker container."
	@echo "It must have access to the project root at ../../"
	@cd ../../ \
		&& python3 ${DIVISION}/$(SUBMITTER)/build/inference/tools/submission/submission-checker.py --input . --submitter $(SUBMITTER) 2>&1 \
		| tee ${DIVISION}/$(SUBMITTER)/results/submission_checker_log.txt

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
