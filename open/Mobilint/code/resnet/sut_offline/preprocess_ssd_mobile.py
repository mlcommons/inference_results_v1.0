import numpy as np
import cv2
import os
import glob
from tqdm import tqdm


def maybe_resize(img, im_height, im_width):
    img = np.array(img, dtype=np.float32)
    if len(img.shape) < 3 or img.shape[2] != 3:
        # some images might be grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (im_width, im_height), interpolation=cv2.INTER_LINEAR)
    
    return img


def preprocess_coco(load_dir_name, save_dir_name):
    filelist = glob.glob(load_dir_name + "*.jpg")
    print("total file number is ", len(filelist))

    for file_path in tqdm(filelist):
        file_name = file_path.split('/')[-1]
        save_path = os.path.join(save_dir_name, file_name[:-4] + '.bin')

        # read and resize
        img = cv2.imread(file_path)
        img = maybe_resize(img, 300, 300)
        
        # normalize image
        img -= 127.5
        img /= 127.5

        # convert to int8
        input_scale = 0.007874015748031496
        img = np.clip(img / input_scale, a_min = -128, a_max = 127)
        img = np.floor(img + 0.5).astype(np.int8)
        # zero padding
        save_img = np.zeros((300, 320, 3), dtype=np.int8)
        save_img[:, :300, :] = img
        # save
        save_img.tofile(save_path)


if __name__ == "__main__":
    load_dir_name = "/data/mlperf-2020/datasets/coco/val2017/"
    save_dir_name = "/data/mlperf-2020/datasets/coco/val2017_ssd_mobile_preprocessed_64/"

    if not os.path.isdir(save_dir_name):
        os.mkdir(save_dir_name)

    preprocess_coco(load_dir_name, save_dir_name)