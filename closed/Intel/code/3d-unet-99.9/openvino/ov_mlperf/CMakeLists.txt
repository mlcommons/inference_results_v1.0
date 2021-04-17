# Copyright (c) 2018 Intel Corporation

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#      http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
cmake_minimum_required (VERSION 2.8)

set (TARGET_NAME "ov_mlperf")

if (APPLE)
    set(CMAKE_MACOSX_RPATH 1)
endif(APPLE)

# Loadgen
if (NOT(LOADGEN_DIR))
   MESSAGE(FATAL_ERROR "LOADGEN_DIR not set. Set -DLOADGEN_DIR=</path/to/loadgen/>")
else ()
    if(EXISTS ${LOADGEN_DIR})
        message(STATUS "Loadgen_dir: ${LOADGEN_DIR}")
    else()
        message(FATAL_ERROR "${LOADGEN_DIR} does not exists. Please check path.")
    endif()
endif()

if (NOT(LOADGEN_LIB_DIR))
    set(LOADGEN_LIB_DIR ${LOADGEN_DIR})
    message(STATUS "LOADGEN_LIB_DIR unspecified. Set to ${LOADGEN_LIB_DIR}.")
endif()

find_library(LOADGEN_LIBRARY NAMES mlperf_loadgen PATHS "${LOADGEN_LIB_DIR}/")

if (NOT LOADGEN_LIBRARY)
    message(FATAL_ERROR "Loadgen library not found. Use -DLOADGEN_LIB_DIR to specify location")
endif()

message(STATUS "LOADGEN_LIBRARY Found: ${LOADGEN_LIBRARY}")    


find_package(InferenceEngine REQUIRED)
find_package(ngraph REQUIRED)
find_package(gflags REQUIRED)

file (GLOB MAIN_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

file (GLOB MAIN_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.h)

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj
source_group ("src" FILES ${MAIN_SRC})
source_group ("include" FILES ${MAIN_HEADERS})


include_directories (${InferenceEngine_INCLUDE_DIRS}
                     ${LOADGEN_DIR})

# Create library file from sources.
add_executable (${TARGET_NAME} ${MAIN_SRC} ${MAIN_HEADERS})

set_target_properties (${TARGET_NAME} PROPERTIES "CMAKE_CXX_FLAGS" "${CMAKE_CXX_FLAGS} -fPIE")

target_link_libraries (${TARGET_NAME} ${InferenceEngine_LIBRARIES} gflags ngraph::ngraph ${LOADGEN_LIBRARY})

if (UNIX)
    set (LIB_DL dl)
    target_link_libraries (${TARGET_NAME} ${LIB_DL} pthread)
endif ()
