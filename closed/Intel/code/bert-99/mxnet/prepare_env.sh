set -x
proxy=${http_proxy}

echo "Install dependencies"
conda install cmake --yes
conda install absl-py --yes

WORKDIR=`pwd`
echo "Install loadgen"
git clone --recurse-submodules https://github.com/mlcommons/inference.git mlperf_inference
cd mlperf_inference
git checkout r1.0 
git log -1
git submodule update --init --recursive
cd loadgen
CFLAGS="-std=c++14" python setup.py install
cd ../..

source /opt/intel/compilers_and_libraries/linux/mkl/bin/mklvars.sh intel64
cd ${WORKDIR}/<this/repo>/closed/Intel/calibration/MXNet/bert
http_proxy=$proxy https_proxy=$proxy bash download_model.sh

cp ${WORKDIR}/mlperf_inference/mlperf.conf ${WORKDIR}/<this/repo>/closed/Intel/code/bert-99/mxnet/
cp ${WORKDIR}/mlperf_inference/language/bert/bert_config.json .
cd ${WORKDIR} 

echo "Install dependencies and convert tensorflow model to MXNet"
echo "Install MXNet, gluon-nlp and tensorflow"
# install MXNet
git clone https://github.com/apache/incubator-mxnet.git
cd incubator-mxnet
git fetch origin pull/20016/head:bert_optimizations
git checkout bert_optimizations
git log -1
git submodule update --init
make -j USE_OPENCV=0 USE_MKLDNN=1 USE_BLAS=mkl USE_PROFILER=0 USE_LAPACK=0 USE_GPERFTOOLS=0 USE_INTEL_PATH=/opt/intel/
cd python && python setup.py install
cd ../..

# install gluonnlp
git clone https://github.com/dmlc/gluon-nlp.git
cd gluon-nlp
git checkout 0d5c61992180f41eab590e74c7b679980f429292
git log -1
http_proxy=$proxy https_proxy=$proxy python setup.py develop
cd ..

# install tensorflow/transformers
conda install tensorflow==1.15.0 --yes
conda install -c conda-forge transformers --yes
conda list|grep transformers --yes

echo "if mxnet cannot find libiomp5.so when importing, please add the path of it to LD_LIBRARY_PATH"
export LD_LIBRARY_PATH=/opt/intel/lib/intel64_lin:$LD_LIBRARY_PATH

echo "Convert tensorflow model to MXNet"

cd ${WORKDIR}/<this/repo>/closed/Intel/calibration/MXNet/bert/
python bert_tf_to_mxnet.py --model=bert_24_1024_16 --tf_model_path=./build/model/  --tf_config_name=bert_config.json --out_dir=./converted_from_tf_to_mxnet/

echo "Build optimization graph pass"
echo "Set mxnet_path to MXNet in graphpass_setup.py file"
sed -i "s#mxnet_path = pathlib.Path.*#mxnet_path = pathlib.Path(\"${WORKDIR}/incubator-mxnet\")#" bertpass_setup.py
http_proxy=$proxy https_proxy=$proxy python bertpass_setup.py install

echo "Quantize model for offline scenario"
export OMP_NUM_THREADS=28
export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0
numactl --physcpubind=0-27 --membind=0 \
      python calibration.py \
        --bert_model=bert_24_1024_16 \
        --model_parameters=./converted_from_tf_to_mxnet/tf_fp32.params \
        --vocab_path=./converted_from_tf_to_mxnet/tf.vocab \
        --max_seq_length=384 \
        --output_dir=./converted_from_tf_to_mxnet/offline_model \
        --num_calib_batches=10 \
        --test_batch_size=10 \
        --scenario=offline

echo "Quantize model for server scenario"
MXNET_DISABLE_ONEDNN_BRGEMM_FC=1 \
MXNET_DISABLE_MKLDNN_FC_U8_FC_OPT=1 \
MXNET_DISABLE_MKLDNN_FC_SUM_OPT=1 \
MXNET_DISABLE_MKLDNN_ASYM_QUANT_FC_OPT=1 \
MXNET_DISABLE_MKLDNN_INTERLEAVED_U8_FC_OPT=0 \
numactl --physcpubind=0-23 --membind=0 \
      python calibration.py \
        --bert_model=bert_24_1024_16 \
        --model_parameters=./converted_from_tf_to_mxnet/tf_fp32.params \
        --vocab_path=./converted_from_tf_to_mxnet/tf.vocab \
        --max_seq_length=384 \
        --output_dir=./converted_from_tf_to_mxnet/server_model \
        --num_calib_batches=10 \
        --test_batch_size=10 \
        --scenario=server

# Copy performance calibration profile as the working performance profile
cp profiles/prof_clx28c.py ${WORKDIR}/<this/repo>/closed/Intel/code/bert-99/mxnet/prof.py

set +x
