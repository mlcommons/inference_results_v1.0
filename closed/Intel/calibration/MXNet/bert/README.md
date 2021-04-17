# Requirments for MXNet BERT
Before you run scripts below make sure you have following packages installed:
- MXNet
- tensorflow==1.15.0
- transformers
- mlperf inference loadgen
- gluon-nlp

Follow README.md in the 'closed/Intel/code/bert-99/mxnet' directory to install them.  
**${WORKDIR}** below is path to cloned dependencies. 

#  Converting TensorFlow model
BERT-Large weights must be converted from TensorFlow model to MXNet compatible format. In order to do that follow steps below:
1. Download TensorFlow model (files downloaded by this script will be placed in `build` folder):  
```bash download_model.sh```

2. Copy `bert_config.json` file from MLPerf repository:  
```cp ${WORKDIR}/mlperf_inference/language/bert/bert_config.json```

3. Run following python script to convert TF to MXNet:  
```python bert_tf_to_mxnet.py --model=bert_24_1024_16 --tf_model_path=./build/model/ --tf_config_name=bert_config.json --out_dir=./converted_from_tf_to_mxnet/```

Compatible with MXNet weights will be put in new `converted_form_tf_to_mxnet` folder named `tf_fp2.params`

#  Setup MXNet's Graph Pass extension
MXNet let user to to use graph pass feature which enables users to write custom model modification strategies without compiling against all of MXNet header files and dependencies. For more information please follow [MXNet Graph Pass](https://github.com/apache/incubator-mxnet/tree/v1.x/example/extensions/lib_pass)  

Install prepared graph pass for BERT-Large model - following script will produce `bertpass.so` dynamic library:  
```python bertpass_setup.py install```

# Quantize MXNet Model
MXNet allow user to quantize model with simple API. For more information about MXNet Quantization API use following link [MXNet Quantization](https://mxnet.apache.org/versions/1.6/api/python/docs/tutorials/performance/backend/mkldnn/mkldnn_quantization.html).  
Script belows run installed graph passes, quantize and calibrate model with validation data.
It's recommended to use different optimizations for Server Scenario and Offline Scenario.

## Server Scenario
```
MXNET_DISABLE_ONEDNN_BRGEMM_FC=1 \
MXNET_DISABLE_MKLDNN_FC_U8_FC_OPT=1 \
numactl --physcpubind=0-27 --membind=0 \
      python calibration.py \
        --bert_model=bert_24_1024_16 \
        --model_parameters=./converted_from_tf_to_mxnet/tf_fp32.params \
        --vocab_path=./converted_from_tf_to_mxnet/tf.vocab \
        --max_seq_length=384 \
        --output_dir=./converted_from_tf_to_mxnet/server_model \
        --num_calib_batches=10 \
        --test_batch_size=10
```
## Offline Scenario
```
MXNET_DISABLE_MKLDNN_FC_U8_FC_OPT=1 \
numactl --physcpubind=0-27 --membind=0 \
      python calibration.py \
        --bert_model=bert_24_1024_16 \
        --model_parameters=./converted_from_tf_to_mxnet/tf_fp32.params \
        --vocab_path=./converted_from_tf_to_mxnet/tf.vocab \
        --max_seq_length=384 \
        --output_dir=./converted_from_tf_to_mxnet/offline_model \
        --num_calib_batches=10 \
        --test_batch_size=10
```
