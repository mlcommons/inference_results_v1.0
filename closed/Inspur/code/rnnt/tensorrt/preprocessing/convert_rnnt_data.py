# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
    Script to preprocess .wav files and convert them to .npy format
    RNNT harness reads in .npy files

    Example command line:
        python3 convert_rnnt_data.py --batch_size 1 --output_dir <path> --cudnn_benchmark --dataset_dir <path> --val_manifest <path>/<name>-wav.json --model_toml configs/rnnt.toml
'''

import argparse
import itertools
import os
import torch
import numpy as np
import torchvision
from tqdm import tqdm
import math
import random
import toml
import sys

sys.path.insert(0, os.path.dirname(__file__))
from helpers import Optimization, print_dict, add_blank_label
from dataset import AudioToTextDataLayer
from preprocessing import AudioPreprocessing


def parse_args():
    parser = argparse.ArgumentParser(description='Jasper')
    parser.add_argument("--dataset_dir", type=str, help='absolute path to dataset folder')
    parser.add_argument("--output_dir", type=str, help='absolute path for generated .npy files folder')
    parser.add_argument("--val_manifest", type=str, help='relative path to evaluation dataset manifest file')
    parser.add_argument("--batch_size", default=1, type=int, help='data batch size')
    parser.add_argument("--fp16", action='store_true', help='use half precision')
    parser.add_argument("--fixed_seq_length", default=512, type=int, help="produce .npy files with fixed sequence length")
    parser.add_argument("--generate_wav_npy", default=True, type=str, help="produce wav .npy files with MAX length")
    parser.add_argument("--fixed_wav_file_length", default=240000, type=int, help="produce wav .npy files with MAX length")
    parser.add_argument("--seed", default=42, type=int, help='seed')
    parser.add_argument("--model_toml", type=str, help='relative model configuration path given dataset folder')
    parser.add_argument("--max_duration", default=None, type=float, help='maximum duration of sequences. if None uses attribute from model configuration file')
    parser.add_argument("--pad_to", default=None, type=int, help="default is pad to value as specified in model configurations. if -1 pad to maximum duration. If > 0 pad batch to next multiple of value")
    return parser.parse_args()


def eval(
        data_layer,
        audio_processor,
        args):

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(args.output_dir + 'fp16'):
        os.makedirs(args.output_dir + "fp16")
    if not os.path.exists(args.output_dir + 'fp32'):
        os.makedirs(args.output_dir + "fp32")
    if not os.path.exists(args.output_dir + 'int32'):
        os.makedirs(args.output_dir + "int32")
    if(args.generate_wav_npy):
        if not os.path.exists(args.output_dir + 'wav_files'):
            os.makedirs(args.output_dir + "wav_files")
        if not os.path.exists(args.output_dir + 'wav_files' + '/int32'):
            os.makedirs(args.output_dir + 'wav_files' + '/int32')
        if not os.path.exists(args.output_dir + 'wav_files' + '/fp32'):
            os.makedirs(args.output_dir + 'wav_files' + '/fp32')
        if not os.path.exists(args.output_dir + 'wav_files' + '/fp16'):
            os.makedirs(args.output_dir + 'wav_files' + '/fp16')

    fixed_seq_length = args.fixed_seq_length
    val_map_filename = args.output_dir + "val_map_" + str(fixed_seq_length) + ".txt"

    file_handle = open(val_map_filename, "w")

    max_seq_length = 0
    for it, data in enumerate(tqdm(data_layer.data_iterator)):
        tensors = []
        for d in data:
            tensors.append(d)

        file_handle.write("RNNT_input_" + str(fixed_seq_length) + "_" + str(it) + "\n")

        if(args.generate_wav_npy):
            t_audio_signal_e, t_a_sig_length_e, t_transcript_e, t_transcript_len_e = tensors

            print("Audio signal = {} dtype = {} shape {} ".format(t_audio_signal_e, t_audio_signal_e.dtype, torch.numel(t_audio_signal_e)))
            print("{} Audio signal length = {}".format(it, t_a_sig_length_e))
            t_audio_signal_e_fp16 = t_audio_signal_e.to(torch.float16)

            if t_a_sig_length_e <= args.fixed_wav_file_length:
                target = torch.zeros(args.fixed_wav_file_length, dtype=torch.float32)
                target[:t_a_sig_length_e] = t_audio_signal_e
                #print("Target = {}".format(target))
                #print("Target num elements = {}".format(torch.numel(target)))
                target_np = target.cpu().numpy()
                file_name = args.output_dir + "wav_files/fp32/" + "RNNT_input_" + str(fixed_seq_length) + "_" + str(it) + ".npy"
                np.save(file_name, target_np)

                target = torch.zeros(args.fixed_wav_file_length, dtype=torch.float16)
                target[:t_a_sig_length_e] = t_audio_signal_e_fp16
                #print("Target = {}".format(target))
                #print("Target num elements = {}".format(torch.numel(target)))
                target_np = target.cpu().numpy()
                file_name = args.output_dir + "wav_files/fp16/" + "RNNT_input_" + str(fixed_seq_length) + "_" + str(it) + ".npy"
                np.save(file_name, target_np)

                t_a_sig_length_e_int32 = t_a_sig_length_e.to(torch.int32)
                t_a_sig_length_e_int32_np = t_a_sig_length_e_int32.cpu().numpy()
                print("Length tensor = {}".format(t_a_sig_length_e_int32_np))
                file_name = args.output_dir + "wav_files/int32/" + "RNNT_input_" + str(fixed_seq_length) + "_" + str(it) + ".npy"
                np.save(file_name, t_a_sig_length_e_int32_np)
            else:
                target = t_audio_signal_e_fp16[:args.fixed_wav_file_length]
                target_np = target.cpu().numpy()
                file_name = args.output_dir + "wav_files/fp32/" + "RNNT_input_" + str(fixed_seq_length) + "_" + str(it) + ".npy"
                np.save(file_name, target_np)

                length_tensor = torch.Tensor([args.fixed_wav_file_length])
                #print("Length_tensor = {}".format(length_tensor))
                t_a_sig_length_e_int32 = length_tensor.to(torch.int32)
                t_a_sig_length_e_int32_np = t_a_sig_length_e_int32.cpu().numpy()
                print("Length tensor = {}".format(t_a_sig_length_e_int32_np))
                file_name = args.output_dir + "wav_files/int32/" + "RNNT_input_" + str(fixed_seq_length) + "_" + str(it) + ".npy"
                np.save(file_name, t_a_sig_length_e_int32_np)

        t_audio_signal_e, t_a_sig_length_e, t_transcript_e, t_transcript_len_e = audio_processor(data)

        seq_length, batch_size, num_features = t_audio_signal_e.size()
        print("Seq length = {} Batch size = {} Features = {}".format(seq_length, batch_size, num_features))
        if seq_length > max_seq_length:
            max_seq_length = seq_length

        t_audio_signal_e_fp16 = t_audio_signal_e.to(torch.float16)
        t_audio_signal_e_fp16 = t_audio_signal_e_fp16.reshape(seq_length, num_features)
        t_audio_signal_e_fp16_np = t_audio_signal_e_fp16.cpu().numpy()

        t_audio_signal_e = t_audio_signal_e.reshape(seq_length, num_features)
        t_audio_signal_e_np = t_audio_signal_e.cpu().numpy()

        t_a_sig_length_e_int32 = t_a_sig_length_e.to(torch.int32)
        t_a_sig_length_e_int32_np = t_a_sig_length_e_int32.cpu().numpy()

        target_np = t_a_sig_length_e_int32_np
        file_name = args.output_dir + "int32/RNNT_input_" + str(fixed_seq_length) + "_" + str(it) + ".npy"
        np.save(file_name, target_np)

        # Generating Fixed size seq_length
        if seq_length <= fixed_seq_length:
            target = torch.zeros(fixed_seq_length, 240, dtype=torch.float16)
            target[:seq_length, :] = t_audio_signal_e_fp16
            target_np = target.cpu().numpy()
            file_name = args.output_dir + "fp16/RNNT_input_" + str(fixed_seq_length) + "_" + str(it) + ".npy"
            np.save(file_name, target_np)

            target = torch.zeros(fixed_seq_length, 240, dtype=torch.float32)
            target[:seq_length, :] = t_audio_signal_e
            target_np = target.cpu().numpy()
            file_name = args.output_dir + "fp32/RNNT_input_" + str(fixed_seq_length) + "_" + str(it) + ".npy"
            np.save(file_name, target_np)
        else:
            target = torch.zeros(fixed_seq_length, 240, dtype=torch.float16)
            target = t_audio_signal_e_fp16[:fixed_seq_length, :]
            target_np = target.cpu().numpy()
            file_name = args.output_dir + "fp16/RNNT_input_" + str(fixed_seq_length) + "_" + str(it) + ".npy"
            np.save(file_name, target_np)

            target = torch.zeros(fixed_seq_length, 240, dtype=torch.float32)
            target = t_audio_signal_e[:fixed_seq_length, :]
            target_np = target.cpu().numpy()
            file_name = args.output_dir + "fp32/RNNT_input_" + str(fixed_seq_length) + "_" + str(it) + ".npy"
            np.save(file_name, target_np)

    print("Max seq length {}".format(max_seq_length))
    file_handle.close()


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.fp16:
        optim_level = Optimization.mxprO3
    else:
        optim_level = Optimization.mxprO0

    model_definition = toml.load(args.model_toml)
    dataset_vocab = model_definition['labels']['labels']
    ctc_vocab = add_blank_label(dataset_vocab)

    val_manifest = args.val_manifest
    featurizer_config = model_definition['input_eval']
    featurizer_config["optimization_level"] = optim_level

    if args.max_duration is not None:
        featurizer_config['max_duration'] = args.max_duration
    if args.pad_to is not None:
        featurizer_config['pad_to'] = args.pad_to if args.pad_to >= 0 else "max"

    data_layer = AudioToTextDataLayer(
        dataset_dir=args.dataset_dir,
        featurizer_config=featurizer_config,
        manifest_filepath=val_manifest,
        labels=dataset_vocab,
        batch_size=args.batch_size,
        pad_to_max=featurizer_config['pad_to'] == "max",
        shuffle=False,
        multi_gpu=False)

    audio_preprocessor = AudioPreprocessing(**featurizer_config)

    audio_preprocessor.eval()

    eval_transforms = torchvision.transforms.Compose([
        lambda xs: [*audio_preprocessor(xs[0:2]), *xs[2:]],
        lambda xs: [xs[0].permute(2, 0, 1), *xs[1:]],
    ])

    eval(
        data_layer=data_layer,
        audio_processor=eval_transforms,
        args=args)


if __name__ == "__main__":
    args = parse_args()

    print_dict(vars(args))

    main(args)
