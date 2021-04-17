import numpy as np
import cv2
import os
import glob
from tqdm import tqdm


def center_crop(img, out_height, out_width):
    height, width, _ = img.shape
    left = int((width - out_width) / 2)
    right = int((width + out_width) / 2)
    top = int((height - out_height) / 2)
    bottom = int((height + out_height) / 2)
    img = img[top:bottom, left:right]
    return img


def resize_with_aspectratio(img, out_height, out_width, scale=87.5, inter_pol=cv2.INTER_LINEAR):
    height, width, _ = img.shape
    new_height = int(100. * out_height / scale)
    new_width = int(100. * out_width / scale)
    if height > width:
        w = new_width
        h = int(new_height * height / width)
    else:
        h = new_height
        w = int(new_width * width / height)
    img = cv2.resize(img, (w, h), interpolation=inter_pol)
    return img


def preprocess_resnet(load_dir_name, save_dir_name):
    filelist = glob.glob(load_dir_name + "*.JPEG")
    print("total file number is ", len(filelist))

    for file_path in tqdm(filelist):
        file_name = file_path.split('/')[-1]
        save_path = os.path.join(save_dir_name, file_name[:-5] + '.bin')

        img = cv2.imread(file_path)
        if len(img.shape) < 3 or img.shape[2] < 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # resize and center crop
        img = resize_with_aspectratio(img, 224, 224)
        img = center_crop(img, 224, 224)
        img = np.asarray(img, dtype=np.float32)

        # normalize image
        means = np.array([123.68, 116.78, 103.94], dtype=np.float32)
        img -= means

        # convert to int8
        input_scale = 1.1894487996739664
        img = np.clip(img / input_scale, a_min = -128, a_max = 127)
        img = np.floor(img + 0.5).astype(np.int8)
        # zero padding
        save_img = np.zeros((224, 256, 3), dtype=np.int8)
        save_img[:, :224, :] = img
        # save
        save_img.tofile(save_path)
    
    # make new val map
    val_map_path = os.path.join(load_dir_name, 'val_map.txt')
    new_val_map_path = os.path.join(save_dir_name, 'val_map.txt')

    with open(val_map_path, 'r') as f:
        lines = f.read().splitlines()
        new_lines = []
        for line in lines:
            filename, label = line.split(' ')
            new_filename = filename[:-5] + '.bin'
            new_line = " ".join([new_filename, label])
            new_lines.append(new_line)

    with open(new_val_map_path, 'w') as f:
        for new_line in new_lines:
            f.write(new_line + '\n')


if __name__ == "__main__":
    load_dir_name = "/data/mlperf-2020/datasets/imagenet/val/"
    save_dir_name = "/data/mlperf-2020/datasets/imagenet/val_resnet_preprocessed_64/"

    if not os.path.isdir(save_dir_name):
        os.mkdir(save_dir_name)

    preprocess_resnet(load_dir_name, save_dir_name)