cmake_minimum_required(VERSION 3.0)
set(TARGET_NAME ov_mlperf)
project(${TARGET_NAME})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIE -fstack-protector-strong")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -fno-operator-names")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -pthread -USE_OPENCV -DBOOST_ERROR_CODE_HEADER_ONLY -DBOOST_NO_CXX11_SCOPED_ENUMS")
set(CMAKE_BUILD_TYPE "Release")

if (CMAKE_BUILD_TYPE STREQUAL "")
    message(STATUS "CMAKE_BUILD_TYPE not defined, 'Release' will be used")
    set(CMAKE_BUILD_TYPE "Release")
endif()

if (NOT(BUILD_DIR))
    message(STATUS "BUILD_DIR not defined. Will build to current directory")
    set(BUILD_DIR ${CMAKE_CURRENT_SOURCE_DIR})
endif()

if (NOT(BIN_FOLDER))
    set(BIN_FOLDER ${BUILD_DIR}/${CMAKE_BUILD_TYPE})
else()
	set(BIN_FOLDER ${BUILD_DIR}/${CMAKE_BUILD_TYPE}_${BIN_FOLDER})
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BIN_FOLDER})

# OpenCV libs
find_package(OpenCV REQUIRED)
find_package(InferenceEngine REQUIRED)
find_package(gflags REQUIRED)
message(STATUS "GFLAGS_INCLUDE_DIR: ${gflags_include_dir}")

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
    find_library(LOADGEN_LIBRARY NAMES mlperf_loadgen PATHS "${LOADGEN_DIR}/")

    if (NOT LOADGEN_LIBRARY)
        message(FATAL_ERROR "Loadgen library not found. Use -DLOADGEN_LIB_DIR to specify location")
    endif()

    message(STATUS "LOADGEN_LIBRARY Found: ${LOADGEN_LIBRARY}" )
    
endif()


# Add BOOST
if (NOT(BOOST_FILESYSTEM_LIB))
    message(FATAL_ERROR "BOOST_FILESYSTEM_LIB required. Please set -DBOOST_FILESYSTEM_LIB=/path/to/libboost_filesystem.so")
endif()

include_directories (
    ${OpenCV_INCLUDE_DIRS}
    ${LOADGEN_DIR}
    ${BOOST_INCLUDE_DIRS}
    ${InferenceEngine_INCLUDE_DIRS}
    ${IE_SRC_DIR}/extension
    ${PYTHON_INCLUDE_DIRS}
)

set(SOURCE_FILES main_ov.cc) #input_flags.h backend_ov.h dataset_ov.h sut_ov.h infer_request_wrap.h main_ov.cc)
add_executable(${TARGET_NAME} ${SOURCE_FILES})

target_link_libraries(${TARGET_NAME}
    ${LOADGEN_LIBRARY}
    ${OpenCV_LIBS}
    ${InferenceEngine_LIBRARIES}
    ${BOOST_FILESYSTEM_LIB}
    gflags
    ${CMAKE_DL_LIBS}
    ${PYTHON_LIBRARIES}
    )
