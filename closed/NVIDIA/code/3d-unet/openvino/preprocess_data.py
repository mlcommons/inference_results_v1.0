#!/usr/bin/env python3
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
# Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
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

"""Preprocess data for 3D-UNet benchmark to npy files."""

import SimpleITK as sitk
import argparse
import json
import numpy as np
import os
import pickle
import shutil
import struct
import sys
from collections import OrderedDict
sys.path.insert(0, os.getcwd())

from code.common import run_command
from batchgenerators.augmentations.utils import pad_nd_image
from batchgenerators.utilities.file_and_folder_operations import subfiles
from nnunet.training.model_restore import load_model_and_checkpoint_files
from nnunet.inference.predict import preprocess_multithreaded


def maybe_mkdir(dir):
    """mkdir the entire path. Do not complain if dir exists."""
    os.makedirs(dir, exist_ok=True)


def copy_BraTS_segmentation_and_convert_labels(in_file, out_file):
    """
    Convert BraTS segmentation labels (nnUnet) and copy file to destination.
    Change [0,1,2,4] labels to [0,2,1,3].
    Used for segmentation only.
    """

    img = sitk.ReadImage(in_file)
    img_npy = sitk.GetArrayFromImage(img)

    uniques = np.unique(img_npy)
    for u in uniques:
        if u not in [0, 1, 2, 4]:
            raise RuntimeError('unexpected label')

    seg_new = np.zeros_like(img_npy)
    seg_new[img_npy == 4] = 3
    seg_new[img_npy == 2] = 1
    seg_new[img_npy == 1] = 2
    img_corr = sitk.GetImageFromArray(seg_new)
    img_corr.CopyInformation(img)
    sitk.WriteImage(img_corr, out_file)


def preprocess_3dunet_raw(data_dir, preprocessed_data_dir):
    """
    Preprocess downloaded BraTS data into raw data folders.
    """

    print("starting preprocessing raw...")

    task_name = "Task043_BraTS2019"
    downloaded_data_dir = os.path.join(data_dir, "BraTS", "MICCAI_BraTS_2019_Data_Training")
    nnUNet_raw_data = os.path.join(preprocessed_data_dir, "brats", "brats_reference_raw")

    target_base = os.path.join(nnUNet_raw_data, task_name)
    target_imagesTr = os.path.join(target_base, "imagesTr")
    target_labelsTr = os.path.join(target_base, "labelsTr")

    maybe_mkdir(target_imagesTr)
    maybe_mkdir(target_labelsTr)

    patient_names = []
    for tpe in ["HGG", "LGG"]:
        cur = os.path.join(downloaded_data_dir, tpe)
        subdirs = [i for i in os.listdir(cur) if os.path.isdir(os.path.join(cur, i))]
        for p in subdirs:
            patdir = os.path.join(cur, p)
            patient_name = tpe + "__" + p
            print("Found patient_name {:}...".format(patient_name))
            patient_names.append(patient_name)
            t1 = os.path.join(patdir, p + "_t1.nii.gz")
            t1c = os.path.join(patdir, p + "_t1ce.nii.gz")
            t2 = os.path.join(patdir, p + "_t2.nii.gz")
            flair = os.path.join(patdir, p + "_flair.nii.gz")
            seg = os.path.join(patdir, p + "_seg.nii.gz")

            assert all([
                os.path.isfile(t1),
                os.path.isfile(t1c),
                os.path.isfile(t2),
                os.path.isfile(flair),
                os.path.isfile(seg)
            ]), "%s" % patient_name

            shutil.copy(t1, os.path.join(target_imagesTr, patient_name + "_0000.nii.gz"))
            shutil.copy(t1c, os.path.join(target_imagesTr, patient_name + "_0001.nii.gz"))
            shutil.copy(t2, os.path.join(target_imagesTr, patient_name + "_0002.nii.gz"))
            shutil.copy(flair, os.path.join(target_imagesTr, patient_name + "_0003.nii.gz"))

            copy_BraTS_segmentation_and_convert_labels(seg, os.path.join(target_labelsTr, patient_name + ".nii.gz"))

    json_dict = OrderedDict()
    json_dict['name'] = "BraTS2019"
    json_dict['description'] = "nothing"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "see BraTS2019"
    json_dict['licence'] = "see BraTS2019 license"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "T1",
        "1": "T1ce",
        "2": "T2",
        "3": "FLAIR"
    }
    json_dict['labels'] = {
        "0": "background",
        "1": "edema",
        "2": "non-enhancing",
        "3": "enhancing",
    }
    json_dict['numTraining'] = len(patient_names)
    json_dict['numTest'] = 0
    json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in
                             patient_names]
    json_dict['test'] = []

    with open(os.path.join(target_base, "dataset.json"), "w") as f:
        json.dump(json_dict, f)


def preprocess_MLPerf(model, checkpoint_name, folds, fp16, list_of_lists, output_filenames, preprocessing_folder, num_threads_preprocessing):
    """
    Helper function to launch multithread to preprocess raw image data to pkl files.
    """
    assert len(list_of_lists) == len(output_filenames)

    print("loading parameters for folds", folds)
    trainer, _ = load_model_and_checkpoint_files(model, folds, fp16=fp16, checkpoint_name=checkpoint_name)

    print("starting preprocessing generator")
    preprocessing = preprocess_multithreaded(trainer, list_of_lists, output_filenames, num_threads_preprocessing, None)
    print("Preprocessing images...")
    all_output_files = []

    for preprocessed in preprocessing:
        output_filename, (d, dct) = preprocessed

        all_output_files.append(output_filename)
        if isinstance(d, str):
            data = np.load(d)
            os.remove(d)
            d = data

        # Pad to the desired full volume
        d = pad_nd_image(d, trainer.patch_size, "constant", None, False, None)

        with open(os.path.join(preprocessing_folder, output_filename + ".pkl"), "wb") as f:
            pickle.dump([d, dct], f)
        f.close()

    return all_output_files


def preprocess_3dunet_ref(model_dir_base, preprocessed_data_dir_base):
    """
    Preprocess raw image data to pickle file.
    """

    print("Preparing for preprocessing data...")

    # Validation set is fold 1
    fold = 1
    validation_fold_file = os.path.join("data_maps", "brats", "val_map.txt")

    # Make sure the model exists
    model_dir = os.path.join(model_dir_base, "3d-unet", "nnUNet", "3d_fullres", "Task043_BraTS2019", "nnUNetTrainerV2__nnUNetPlansv2.mlperf.1")
    model_path = os.path.join(model_dir, "plans.pkl")
    assert os.path.isfile(model_path), "Cannot find the model file {:}!".format(model_path)
    checkpoint_name = "model_final_checkpoint"

    # Other settings
    fp16 = False
    num_threads_preprocessing = 12
    raw_data_dir = os.path.join(preprocessed_data_dir_base, "brats", "brats_reference_raw", "Task043_BraTS2019", "imagesTr")
    preprocessed_data_dir = os.path.join(preprocessed_data_dir_base, "brats", "brats_reference_preprocessed")

    # Open list containing validation images from specific fold (e.g. 1)
    validation_files = []
    with open(validation_fold_file) as f:
        for line in f:
            validation_files.append(line.rstrip())

    # Create output and preprocessed directory
    if not os.path.isdir(preprocessed_data_dir):
        os.makedirs(preprocessed_data_dir)

    # Create list of images locations (i.e. 4 images per case => 4 modalities)
    all_files = subfiles(raw_data_dir, suffix=".nii.gz", join=False, sort=True)
    list_of_lists = [[os.path.join(raw_data_dir, i) for i in all_files if i[:len(j)].startswith(j) and
                      len(i) == (len(j) + 12)] for j in validation_files]

    # Preprocess images, returns filenames list
    # This runs in multiprocess
    print("Actually preprocessing data...")
    preprocessed_files = preprocess_MLPerf(model_dir, checkpoint_name, fold, fp16, list_of_lists,
                                           validation_files, preprocessed_data_dir, num_threads_preprocessing)

    # Save list of pkl file paths to pkl file.
    print("Saving metadata of the preprocessed data...")
    with open(os.path.join(preprocessed_data_dir, "preprocessed_files.pkl"), "wb") as f:
        pickle.dump(preprocessed_files, f)

def preprocess_3dunet_npy_inner(base_dir):
    """
    Convert preprocessed pickle files into npy based on data types.
    """
    reference_preprocessed_dir = os.path.join(base_dir, "brats_reference_preprocessed")
    npy_preprocessed_dir = os.path.join(base_dir, "brats_npy")

    print("Loading file names...")
    with open(os.path.join(reference_preprocessed_dir, "preprocessed_files.pkl"), "rb") as f:
        preprocessed_files = pickle.load(f)

    print("Converting data...")
    fp32_dir = os.path.join(npy_preprocessed_dir, "fp32")
    maybe_mkdir(fp32_dir)

    for file in preprocessed_files:
        print("Converting {:}".format(file))
        with open(os.path.join(reference_preprocessed_dir, file + ".pkl"), "rb") as f:
            d, _ = pickle.load(f)
        assert d.shape == (4, 224, 224, 160), "Expecting shape (4, 224, 224, 160) but got {:}".format(d.shape)
        np.save(os.path.join(fp32_dir, file + ".npy"), d.astype(np.float32))

def preprocess_3dunet_npy(preprocessed_data_dir):
    """Convert preprocessed val pickle files into npy based on data types."""
    print("Converting validation data to npy files...")
    preprocess_3dunet_npy_inner(os.path.join(preprocessed_data_dir, "brats"))

def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "--data_dir", "-d",
        help="Directory containing the input data.",
        default="build/data"
    )
    parser.add_argument(
        "--model_dir", "-m",
        help="Directory containing the models.",
        default="build/models"
    )
    parser.add_argument(
        "--preprocessed_data_dir", "-o",
        help="Output directory for the preprocessed data.",
        default="build/preprocessed_data"
    )
    args = parser.parse_args()
    data_dir = args.data_dir
    model_dir = args.model_dir
    preprocessed_data_dir = args.preprocessed_data_dir

    preprocess_3dunet_raw(data_dir, preprocessed_data_dir)
    preprocess_3dunet_ref(model_dir, preprocessed_data_dir)
    preprocess_3dunet_npy(preprocessed_data_dir)

    print("Done!")


if __name__ == '__main__':
    main()
