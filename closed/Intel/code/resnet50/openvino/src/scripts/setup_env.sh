
# Edit paths as appropriate

OPENVINO_LIBRARIES=</path/to/openvino>/bin/intel64/Release
OMP_LIBRARY=</path/to/openvino>/inference-engine/temp/omp/lib
OPENCV_LIBRARIES=</path/to/openvino>/inference-engine/temp/opencv_<x>.<y>.<x>_ubuntu20/opencv/lib
BOOST_LIBRARIES=</path/to/boost/boost_1_72_0/stage/lib
GFLAGS_LIBRARIES=</path/to/gflags-libraries>

export LD_LIBRARY_PATH=${OPENVINO_LIBRARIES}:${OMP_LIBRARY}:${OPENCV_LIBRARIES}:${BOOST_LIBRARIES}:${GFLAGS_LIBRARIES}

