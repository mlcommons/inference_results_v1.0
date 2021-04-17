  #set -eo pipefail  
  set -x

  echo "Install dependencies" 
  pip install sklearn onnx tqdm lark-parser
  pip install -e git+https://github.com/mlperf/logging@0.7.0-rc2#egg=mlperf-logging
  conda install ninja pyyaml setuptools cmake cffi typing --yes
  conda install intel-openmp mkl mkl-include numpy --no-update-deps --yes
  conda install -c conda-forge gperftools --yes
  conda install jemalloc=5.0.1 --yes
  pip install opencv-python absl-py opencv-python-headless

  echo "Install libraries"
  WORKDIR=`pwd`
  mkdir $WORKDIR/local
  export install_dir=$WORKDIR/local
  cd $WORKDIR && mkdir third_party
  wget https://ftp.osuosl.org/pub/xiph/releases/flac/flac-1.3.2.tar.xz -O third_party/flac-1.3.2.tar.xz
  cd third_party && tar xf flac-1.3.2.tar.xz && cd flac-1.3.2
  ./configure --prefix=$install_dir && make && make install

  cd $WORKDIR
  wget https://sourceforge.net/projects/sox/files/sox/14.4.2/sox-14.4.2.tar.gz -O third_party/sox-14.4.2.tar.gz
  cd third_party && tar zxf sox-14.4.2.tar.gz && cd sox-14.4.2
  LDFLAGS="-L${install_dir}/lib" CFLAGS="-I${install_dir}/include" ./configure --prefix=$install_dir --with-flac && make &&    make install

  cd $WORKDIR
  wget http://www.mega-nerd.com/libsndfile/files/libsndfile-1.0.28.tar.gz -O third_party/libsndfile-1.0.28.tar.gz
  cd third_party && tar zxf libsndfile-1.0.28.tar.gz && cd libsndfile-1.0.28
  ./configure --prefix=$install_dir && make && make install

  echo "Install pytorch/ipex"
  export LD_LIBRARY_PATH=$WORKDIR/local/lib:$LD_LIBRARY_PATH

  cd $WORKDIR
  git clone https://github.com/pytorch/pytorch.git
  cd pytorch
  git checkout v1.5.0-rc3
  git log -1
  git submodule sync
  git submodule update --init --recursive
  CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
  pip install -r requirements.txt

  cd $WORKDIR
  git clone https://github.com/intel/intel-extension-for-pytorch.git
  cd intel-extension-for-pytorch
  git checkout mlperf/inference
  git log -1
  git submodule sync
  git submodule update --init --recursive
  pip install lark-parser hypothesis
  cp torch_patches/dpcpp-v1.5-rc3.patch ../pytorch/
  cd ../pytorch
  git apply dpcpp-v1.5-rc3.patch
  python setup.py install

  cd ../intel-extension-for-pytorch
  python setup.py install

  cd $WORKDIR
  git clone https://github.com/pytorch/vision
  cd vision
  git checkout 85b8fbfd31e9324e64e24ca25410284ef238bcb3
  python setup.py install

  echo "Install loadgen"
  cd $WORKDIR
  git clone https://github.com/mlcommons/inference.git
  cd inference && git checkout r1.0
  git log -1
  git submodule sync && git submodule update --init --recursive
  cd loadgen && CFLAGS="-std=c++14" python setup.py install
  cp ${WORKDIR}/inference/mlperf.conf ${WORKDIR}/<this/repo>/closed/Intel/code/rnnt/. 

  echo "Install dependencies for pytorch_SUT.py"
  pip install toml unidecode inflect librosa
  set +x
