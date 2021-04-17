#!/usr/bin/env python3
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

import os
import sys
sys.path.insert(0, os.getcwd())

import argparse
from argparse import Namespace
import json
import numpy as np
import shutil

from code.common import logging, run_command
from code.rnnt.tensorrt.preprocessing.convert_rnnt_data import main as convert_rnnt_data_main


def to_absolute_path(path):
    assert(len(path) > 0)

    if path[0] == "/":
        return path
    return os.path.join(os.getcwd(), path)


def flac_to_wav(absolute_data_dir, librispeech_path, src, dest):
    wav_file_path = os.path.join(librispeech_path, dest)
    manifest_path = os.path.join(librispeech_path, dest + ".json")

    script_cmd = "cd build/inference/speech_recognition/rnnt && python3 pytorch/utils/convert_librispeech.py --input_dir {:} --dest_dir {:} --output_json {:}".format(
        os.path.join(absolute_data_dir, "LibriSpeech", src),
        wav_file_path,
        manifest_path
    )
    run_command(script_cmd)


def preprocess_rnnt(data_dir, preprocessed_data_dir):
    # Use the flac->wav and manifest generation script in the reference repo.
    logging.info("Updating reference repo for convert_librispeech.py script...")
    run_command("make clone_loadgen")

    absolute_data_dir = to_absolute_path(data_dir)
    absolute_preproc_data_dir = to_absolute_path(preprocessed_data_dir)
    librispeech_path = os.path.join(absolute_preproc_data_dir, "LibriSpeech")

    logging.info("Converting flac -> wav and generating manifest.json for test set...")
    flac_to_wav(absolute_data_dir, librispeech_path, "dev-clean", "dev-clean-wav")

    logging.info("Converting wav files to npy files for test set...")
    npy_out_path = os.path.join(absolute_preproc_data_dir, "rnnt_dev_clean_512")
    wav_out_path = os.path.join(absolute_preproc_data_dir, "rnnt_dev_clean_500_raw")
    args = Namespace(
        dataset_dir=librispeech_path + "/",
        output_dir=npy_out_path + "/",
        val_manifest=os.path.join(librispeech_path, "dev-clean-wav.json"),
        batch_size=1,
        fp16=False,
        fixed_seq_length=512,
        generate_wav_npy=True,
        fixed_wav_file_length=240000,
        seed=42,
        model_toml="code/rnnt/tensorrt/preprocessing/configs/rnnt.toml",
        max_duration=15.0,
        pad_to=0
    )
    convert_rnnt_data_main(args)
    shutil.move(os.path.join(npy_out_path, "wav_files"), wav_out_path)

    # Calibration set: 500 sequences selected from train-clean-100
    calibration_file = "build/inference/calibration/LibriSpeech/calibration_files.txt"

    # train-clean-100 is very large, but we only care about the ones in the calibration set
    # Convert the .wav file names to the corresponding .flac files, then transfer the files to a temporary directory
    logging.info("Building calibration set...")
    with open(calibration_file) as f:
        calibration_wavs = f.read().split("\n")

    def wav_to_flac(wav):
        p = wav.split("/")
        p[0] = "train-clean-100"
        p[-1] = p[-1].split(".")[0] + ".flac"
        return p

    calibration_flacs = [wav_to_flac(x) for x in calibration_wavs if len(x) > 0]
    calib_dir = "calib_flacs"

    seen_transcripts = set()

    for flac in calibration_flacs:
        new_dir = flac[:-1]
        new_dir[0] = calib_dir
        assert(len(new_dir) == 3)

        new_dir_path = os.path.join(absolute_data_dir, "LibriSpeech", *new_dir)
        os.makedirs(new_dir_path, exist_ok=True)

        flac_path = os.path.join(absolute_data_dir, "LibriSpeech", *flac)
        new_flac_path = os.path.join(new_dir_path, flac[-1])
        logging.info(flac_path + " -> " + new_flac_path)
        shutil.copyfile(flac_path, new_flac_path)

        trans_file = "{:}-{:}.trans.txt".format(new_dir[1], new_dir[2])
        trans_file_src_path = os.path.join(absolute_data_dir, "LibriSpeech", *flac[:-1], trans_file)
        trans_file_dst_path = os.path.join(new_dir_path, trans_file)

        # Extract transcript for this sample flac
        flac_id = flac[-1].split(".")[0]
        flac_transcript = None
        with open(trans_file_src_path) as transcript_f:
            transcript = transcript_f.read().split("\n")
            for line in transcript:
                if line.startswith(flac_id):
                    flac_transcript = line

        if flac_transcript is None:
            raise ValueError("Invalid flac ID: {:} does not exist in {:}".format(flac_id, trans_file_src_path))

        # Update transcript
        if trans_file in seen_transcripts:
            f = open(trans_file_dst_path, 'a')
        else:
            f = open(trans_file_dst_path, 'w')
            seen_transcripts.add(trans_file)
        f.write(flac_transcript + "\n")
        f.close()

    logging.info("Converting flac -> wav and generating manifest.json for calibration set...")
    flac_to_wav(absolute_data_dir, librispeech_path, calib_dir, "train-clean-100-wav")

    logging.info("Converting wav files to npy files for calibration set...")
    npy_out_path = os.path.join(absolute_preproc_data_dir, "rnnt_train_clean_512_fp32")
    wav_out_path = os.path.join(absolute_preproc_data_dir, "rnnt_train_clean_512_wav")
    args = Namespace(
        dataset_dir=librispeech_path + "/",
        output_dir=npy_out_path + "/",
        val_manifest=os.path.join(librispeech_path, "train-clean-100-wav.json"),
        batch_size=1,
        fp16=False,
        fixed_seq_length=512,
        generate_wav_npy=True,
        fixed_wav_file_length=240000,
        seed=42,
        model_toml="code/rnnt/tensorrt/preprocessing/configs/rnnt.toml",
        max_duration=15.0,
        pad_to=0
    )
    convert_rnnt_data_main(args)
    shutil.move(os.path.join(npy_out_path, "wav_files"), wav_out_path)

    data_map_dir = to_absolute_path("data_maps/rnnt_train_clean_512")
    os.makedirs(data_map_dir, exist_ok=True)

    data_map_path = os.path.join(data_map_dir, "val_map.txt")
    shutil.copyfile(os.path.join(npy_out_path, "val_map_512.txt"), data_map_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", "-d",
        help="Specifies the directory containing the input data.",
        default="build/data"
    )
    parser.add_argument(
        "--preprocessed_data_dir", "-o",
        help="Specifies the output directory for the preprocessed data.",
        default="build/preprocessed_data"
    )
    args = parser.parse_args()
    data_dir = args.data_dir
    preprocessed_data_dir = args.preprocessed_data_dir

    preprocess_rnnt(data_dir, preprocessed_data_dir)

    print("Done!")


if __name__ == '__main__':
    main()
