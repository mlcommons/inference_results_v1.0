#!/bin/bash

BUILD_DIR=build
DATA_DIR=${BUILD_DIR}/data
MODEL_DIR=${BUILD_DIR}/model
RESULT_DIR=${BUILD_DIR}/result

if [ ! -d ${BUILD_DIR} ]; then
    mkdir ${BUILD_DIR};
fi
if [ ! -d ${DATA_DIR} ]; then
    mkdir ${DATA_DIR};
fi
if [ ! -d ${MODEL_DIR} ]; then
    mkdir ${MODEL_DIR};
fi
if [ ! -d ${RESULT_DIR} ]; then
    mkdir ${RESULT_DIR};
fi


if [ ! -f ${DATA_DIR}/dev-v1.1.json ]; then
    wget -O ${DATA_DIR}/dev-v1.1.json https://github.com/rajpurkar/SQuAD-explorer/blob/master/dataset/dev-v1.1.json?raw=true
fi
if [ ! -f ${DATA_DIR}/evaluate-v1.1.py ]; then
    wget -O ${DATA_DIR}/evaluate-v1.1.py https://github.com/allenai/bi-att-flow/raw/master/squad/evaluate-v1.1.py
fi
if [ ! -f ${MODEL_DIR}/vocab.txt ]; then
    wget -O ${MODEL_DIR}/vocab.txt https://zenodo.org/record/3733868/files/vocab.txt?download=1
fi
if [ ! -f ${MODEL_DIR}/model.pb ]; then
    wget -O ${MODEL_DIR}/model.pb https://zenodo.org/record/3939747/files/model.pb?download=1
fi