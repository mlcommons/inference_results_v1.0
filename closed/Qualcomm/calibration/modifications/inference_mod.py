import json
import logging
import os
import time

import cv2
import numpy as np
import torch

import inference
from inference.vision.classification_and_detection.python.coco import Coco
from inference.vision.classification_and_detection.python.coco import log

class CocoMod(Coco):
    def __init__(self, data_path, image_list, name, use_cache=0, image_size=None,
                 image_format="NHWC", pre_process=None, count=None, cache_dir=None,use_label_map=False):
        self.image_size = image_size
        self.image_list = []
        self.label_list = []
        self.image_ids = []
        self.image_sizes = []
        self.count = count
        self.use_cache = use_cache
        self.data_path = data_path
        self.pre_process = pre_process
        self.use_label_map=use_label_map
        if not cache_dir:
            cache_dir = os.getcwd()
        self.cache_dir = os.path.join(cache_dir, "preprocessed", name, image_format)
        # input images are in HWC
        self.need_transpose = True if image_format == "NCHW" else False
        not_found = 0 
        empty_80catageories = 0
        self.annotation_file = image_list
        if self.use_label_map:
            # for pytorch
            label_map = {}
            with open(self.annotation_file) as fin:
                annotations = json.load(fin)
            for cnt, cat in enumerate(annotations["categories"]):
                label_map[cat["id"]] = cnt + 1
        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
        start = time.time()
        images = {}
        with open(image_list, "r") as f:
            coco = json.load(f)
        for i in coco["images"]:
            images[i["id"]] = {"file_name": i["file_name"],
                               "height": i["height"],
                               "width": i["width"],
                               "bbox": [],
                               "category": []}
        for a in coco["annotations"]:
            i = images.get(a["image_id"])
            if i is None:
                continue
            catagory_ids = label_map[a.get("category_id")] if self.use_label_map else a.get("category_id")
            i["category"].append(catagory_ids)
            i["bbox"].append(a.get("bbox"))
        for image_id, img in images.items():
            #image_name = os.path.join("val2017", img["file_name"])
            image_name = os.path.join(img["file_name"])
            src = os.path.join(data_path, image_name)
            if not os.path.exists(src):
                # if the image does not exists ignore it
                not_found += 1
                continue
            if len(img["category"])==0 and self.use_label_map: 
                #if an image doesn't have any of the 81 categories in it    
                empty_80catageories += 1 #should be 48 images - thus the validation sert has 4952 images
                continue 
            if self.use_cache:
                os.makedirs(os.path.dirname(os.path.join(self.cache_dir, image_name)), exist_ok=True)
            dst = os.path.join(self.cache_dir, image_name)
            if self.use_cache and not os.path.exists(dst + ".npy"):
                # cache a preprocessed version of the image
                img_org = cv2.imread(src)
                processed = self.pre_process(img_org, need_transpose=self.need_transpose, dims=self.image_size)
                np.save(dst, processed)
            self.image_ids.append(image_id)
            self.image_list.append(image_name)
            self.image_sizes.append((img["height"], img["width"]))
            self.label_list.append((img["category"], img["bbox"]))
            # limit the dataset if requested
            if self.count and len(self.image_list) >= self.count:
                break
        time_taken = time.time() - start
        if not self.image_list:
            log.error("no images in image list found")
            raise ValueError("no images in image list found")
        if not_found > 0:
            log.info("reduced image list, %d images not found", not_found)
        if empty_80catageories > 0:
            log.info("reduced image list, %d images without any of the 80 categories", empty_80catageories)
        log.info("loaded {} images, cache={}, took={:.1f}sec".format(
            len(self.image_list), use_cache, time_taken))
        self.label_list = np.array(self.label_list)

    def get_item(self, nr):
        """Get image by number in the list."""
        if self.use_cache:
            dst = os.path.join(self.cache_dir, self.image_list[nr])
            img = np.load(dst + ".npy")
            return img, self.label_list[nr]
        else:
            src = self.get_item_loc(nr)
            img_org = cv2.imread(src)
            processed = self.pre_process(img_org, need_transpose=self.need_transpose, dims=self.image_size)
            processed = torch.Tensor(processed).unsqueeze(0)
            return processed, self.label_list[nr]

    def get_samples(self, id_list):
        data = [self.image_list_inmemory[id] for id in id_list]
        labels = self.label_list[id_list]
        return data, labels

