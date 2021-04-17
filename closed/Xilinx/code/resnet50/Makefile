## Copyright 2019 Xilinx Inc.
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.


####
# Set these paths before build
###

MLP_INFER_ROOT=../mlperf-inference
RT_ENGINE=../rt-engine

## Vitis-AI ROOT folder
ifeq "$(origin VAI_ALVEO_ROOT)" "undefined"
endif
## MLPerf Inference ROOT folder
ifeq "$(origin MLP_INFER_ROOT)" "undefined"
endif

PWD := $(shell pwd)
###
# Includes
###
# Common
os=$(shell lsb_release -a | grep "Distributor ID" | sed 's/^.*:\s*//')
os_version=$(shell lsb_release -a | grep "Release" | sed 's/^.*:\s*//')
arch=$(shell uname -p)
target_info=$(os).$(os_version).$(arch).Release
INCLUDE := -I $(CONDA_PREFIX)/include
INCLUDE += -I ${MLP_INFER_ROOT}/loadgen -I /opt/xilinx/xrt/include

install_prefix_default=$(HOME)/.local/Ubuntu.18.04.x86_64
ifndef XILINX_VART
	# setting default path
	XILINX_VART=${install_prefix_default}.Release
endif

#INCLUDE += -I $(HOME)/.local/$(target_info)/include
INCLUDE_R := -I ${RT_ENGINE}/runner/src -I ${RT_ENGINE}/controller/src -I ${RT_ENGINE}/device/src  -I$(XILINX_VART)/include

###
# Libraries
###
LIB_PATHS := -L $(CONDA_PREFIX)/lib
LIB_PATHS += -L $(MLP_INFER_ROOT)/loadgen/build
LIB_PATHS += -L /opt/xilinx/xrt/lib
LIB_PATHS += -L $(HOME)/.local/$(target_info)/lib

#LIB_PATHS_R += -L $(HOME)/.local/Ubuntu.16.04.x86_64.Debug/lib
#LIB_PATHS_R := -L ${RT_ENGINE}/build

# Common
#
LIBS := -lmlperf_loadgen -lpthread
LIBS += -lopencv_imgcodecs -lopencv_core -lopencv_highgui -lopencv_imgproc
#LIBS += -lvart-runner -lxir -lvart-mem-manager -lglog
LIBS += -lxilinxopencl -lxrt_core -lglog -ljson-c -lrt-engine -lxir -lvart-runner

###
# Compiler
###
CXX := g++
# Flags
CXXOPTS :=-O3 -std=c++17
#CXXOPTS :=-O0 -g -ggdb -std=c++17

###
# Make Targets
###
INCLUDE += $(INCLUDE_R)
LIB_PATHS += $(LIB_PATHS_R)
EXE := app.exe

mlperf_app: $(EXE)

$(EXE): app.cc
	@echo -e "make: $<"
	$(CXX) $(CXXOPTS) $< -o $@ $(INCLUDE) $(LIB_PATHS) $(LIBS)

clean:
	@echo -e "make: $@"
	@$(RM) *.exe
