  echo "Install dependency packages"
  pip install sklearn onnx tqdm lark-parser
  pip install -e git+https://github.com/mlperf/logging@0.7.0-rc2#egg=mlperf-logging
  conda install ninja pyyaml setuptools cmake cffi typing --yes
  conda install intel-openmp mkl mkl-include numpy --no-update-deps --yes
  pip install opencv-python  absl-py opencv-python-headless

  #conda install -c conda-forge gperftools       # This is for tcmalloc
  wget https://github.com/gperftools/gperftools/archive/gperftools-2.8.tar.gz
  tar -xvzf gperftools-2.8.tar.gz
  sudo apt-get install  autoconf automake
  cd gperftools-gperftools-2.8
  export install_dir=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
  ./autogen.sh
  ./configure --prefix=$install_dir
  make && make install
  cd ..

  echo "Clone source code and Install"
  echo "Install PyTorch and Intel Extension for PyTorch"

  WORKDIR=`pwd`
  # clone PyTorch
  git clone https://github.com/pytorch/pytorch.git
  cd pytorch && git checkout tags/v1.5.0-rc3 -b v1.5-rc3
  git log -1
  git submodule sync && git submodule update --init --recursive
  cd ..

  # clone Intel Extension for PyTorch
  git clone https://github.com/intel/intel-extension-for-pytorch
  cd intel-extension-for-pytorch
  git checkout mlperf/inference
  git checkout 8f9402e180da8a8a60432c54d390320bac43fb11
  git log -1
  git submodule sync && git submodule update --init --recursive

  # install PyTorch
  cd ${WORKDIR}/pytorch
  cp ${WORKDIR}/intel-extension-for-pytorch/torch_patches/0001-enable-Intel-Extension-for-CPU-enable-CCL-backend.patch .
  patch -p1 < 0001-enable-Intel-Extension-for-CPU-enable-CCL-backend.patch
  python setup.py install

  # install Intel Extension for PyTorch
  cd ${WORKDIR}/intel-extension-for-pytorch
  python setup.py install
  cd ..

  echo "Install loadgen"
  git clone https://github.com/mlcommons/inference.git
  cd inference && git checkout r1.0 
  git log -1
  git submodule update --init --recursive
  cd loadgen
  CFLAGS="-std=c++14" python setup.py install
  cd ../..
  cp ${WORKDIR}/inference/mlperf.conf ${WORKDIR}/<path/to/this/repo>/closed/Intel/code/dlrm-99.9/pytorch/.
