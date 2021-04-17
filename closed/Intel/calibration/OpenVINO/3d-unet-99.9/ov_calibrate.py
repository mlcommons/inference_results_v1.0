#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: EPL-2.0
#

# OpenVINO 2021.2
# export PYTHONPATH=$INTEL_OPENVINO_DIR/deployment_tools/tools/post_training_optimization_toolkit/:$PYTHONPATH

import os
import sys
import numpy as np
import json
import cv2 as cv
import glob
from addict import Dict
from math import ceil
import SimpleITK as sitk
from scipy.special import softmax
from medpy import metric as medpy_metric

from compression.graph import load_model, save_model
from compression.api import Metric, DataLoader
from compression.engines.ie_engine import IEEngine
from compression.pipeline.initializer import create_pipeline

import argparse

sys.path.insert(0, os.path.join(os.getcwd(), "nnUnet"))

from brats_QSL import get_brats_QSL
from nnunet.evaluation.region_based_evaluation import evaluate_regions, get_brats_regions, create_region_from_mask
from nnunet.preprocessing.cropping import ImageCropper


parser = argparse.ArgumentParser(
    description="Quantizes an OpenVINO model to INT8.",
    add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--model", default="build/model/model.xml", help="XML file for OpenVINO to calibrate.")
parser.add_argument("--model_name", default="3d_unet_i8.xml", help="OpenVINO model name.")
parser.add_argument("--preprocessed_data_dir", default="build/calibration", help="Path to preprocessed data.")
parser.add_argument("--int8_directory", default="build/model/calibrated", help="Directory to save calibrated OpenVINO model.")

args = parser.parse_args()


class MyDataLoader(DataLoader):
    def __init__(self, config):
        super().__init__(config)

        self.qsl = get_brats_QSL(config["preprocessed_data_dir"])

        print("Calibrating OpenVINO model")
        print("There are {:,} samples in the dataset ".format(self.qsl.count))

    def __len__(self):
        return self.qsl.count

    def __getitem__(self, item):
        self.qsl.load_query_samples([item])
        data = self.qsl.get_features(item)[np.newaxis, ...]
        return (item, data), data


class MyMetric(Metric):
    def __init__(self):
        super().__init__()
        self.name = "Dice score"
        self._values = []
        self.round = 1

    @property
    def value(self):
        return { self.name: [self._values[-1]] }

    @property
    def avg_value(self):
        value = np.ravel(self._values).mean()
        self.round += 1
        return { self.name: value }

    def update(self, outputs, labels):
        self._values.append(0.9)

    def reset(self):
        self._values = []

    @property
    def higher_better(self):
        return True

    def get_attributes(self):
        return { self.name: {"direction": "higher-better", "type": ""} }

model_config = Dict({
    "model_name": args.model_name,
    "model": args.model,
    "weights": os.path.splitext(args.model)[0] + '.bin'
})

engine_config = Dict({
    "device": "CPU",
    "stat_requests_number": 4,
    "eval_requests_number": 4
})

dataset_config = Dict({
    "preprocessed_data_dir": args.preprocessed_data_dir,
})

algorithms = [
    {
        'name': 'DefaultQuantization',
        'params': {
            'target_device': 'CPU',
            'preset': 'performance',
            'stat_subset_size': 300
        }
    }
]

model = load_model(model_config)

data_loader = MyDataLoader(dataset_config)
metric = MyMetric()

loss = None
engine = IEEngine(engine_config, data_loader, metric)
pipeline = create_pipeline(algorithms, engine)

compressed_model = pipeline.run(model)
save_model(compressed_model, args.int8_directory)

print('Calibrated model successfully saved to: {}'.format(args.int8_directory))