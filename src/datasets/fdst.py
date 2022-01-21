# This file contains the dataloader for the Fudan-ShanghaiTech dataset
# https://github.com/sweetyy83/Lstn_fdst_dataset

import os
import torch
import random
import json
from glob import glob
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torchvision.transforms.functional as F
import numpy as np


def gen_discrete_map(im_height, im_width, points):
    """
    func: generate the discrete map.
    points: [num_gt, 2], for each row: [width, height]
    """
    discrete_map = np.zeros([im_height, im_width], dtype=np.float32)
    h, w = discrete_map.shape[:2]
    num_gt = points.shape[0]
    if num_gt == 0:
        return discrete_map

    # fast create discrete map
    points_np = np.array(points).round().astype(int)
    p_h = np.minimum(points_np[:, 1], np.array([h - 1] * num_gt).astype(int))
    p_w = np.minimum(points_np[:, 0], np.array([w - 1] * num_gt).astype(int))
    p_index = torch.from_numpy(p_h * im_width + p_w)
    discrete_map = torch.zeros(im_width * im_height)\
        .scatter_add_(0, index=p_index, src=torch.ones(im_width * im_height)).view(im_height, im_width).numpy()

    assert np.sum(discrete_map) == num_gt
    return discrete_map


class FDST(Dataset):

    def __init__(self, root_path, training=True, sequence_len=5, crop_size=512, crop_origin_x=600, crop_origin_y=0, downsample_ratio=2, stride=1):
        """
        Constructor of FDST dataset loader
        :param root_path: path to the root directory of the dataset
        :param training: indicates, whether we want to load the data from the train_data or ther test_data subdirectory
        :param sequence_len: number of images, that are contained in one item
        """
        self.data_path = os.path.join(root_path, "train_data" if training else "test_data")
        self.training = training
        self.sequence_len = sequence_len
        self.c_size = crop_size
        self.crop_origin_x = crop_origin_x
        self.crop_origin_y = crop_origin_y
        self.d_ratio = downsample_ratio
        self.stride = stride
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.item_id_dict = {}
        self.index_inputs()

    def index_inputs(self):
        top_i = 0
        overall_length = self.sequence_len * self.stride

        dir_items = os.listdir(self.data_path)      # GET ALL ITEMS IN DIRECTORY
        for name in dir_items:
            item_path = os.path.join(self.data_path, name)
            if os.path.isdir(item_path):                        # IF THE ITEM IS A DIRECTORY
                items = glob(os.path.join(item_path, '*.jpg'))
                items.sort()

                for i, item in enumerate(items):
                    if i >= overall_length:
                        seq_start, seq_end = self.get_seq_bounds(i)
                        self.item_id_dict[top_i] = items[seq_start:seq_end:self.stride]
                        top_i += 1

    def get_seq_bounds(self, i):
        overall_length = self.sequence_len * self.stride
        seq_start = i - overall_length + self.stride
        seq_end = i + 1
        return seq_start, seq_end

    def __len__(self):
        return len(self.item_id_dict.keys())

    def __getitem__(self, idx):
        images = [Image.open(img_path).convert('RGB') for img_path in self.item_id_dict[idx]]
        keypoints = self.load_keypoints(self.item_id_dict[idx][-1])

        h, w, imgs, keypoints = self.crop(images, keypoints)

        gt_discrete = gen_discrete_map(h, w, keypoints)
        down_w = w // self.d_ratio
        down_h = h // self.d_ratio

        gt_discrete = gt_discrete.reshape([down_h, self.d_ratio, down_w, self.d_ratio]).sum(axis=(1, 3))
        assert np.sum(gt_discrete) == len(keypoints)

        if self.training:
            imgs, keypoints, gt_discrete = self.augment(imgs, keypoints, gt_discrete)

        imgs_transformed = [self.trans(img) for img in imgs]
        imgs_tensor = torch.stack(imgs_transformed)
        return imgs_tensor, torch.from_numpy(keypoints.copy()).float(), torch.from_numpy(gt_discrete.copy()).float()

    def crop(self, imgs, keypoints):
        img_w, img_h = imgs[0].size
        st_size = 1.0 * min(img_w, img_h)
        assert st_size >= self.c_size

        if self.training:
            crop_size = int(np.random.uniform(self.c_size/4, min(img_w, img_h)))  # RANDOMLY RESIZE THE IMAGE
            res_h = img_h - crop_size
            res_w = img_w - crop_size
            i = random.randint(0, res_h)
            j = random.randint(0, res_w)
            h, w = crop_size, crop_size     # TODO -> REFACTOR THIS MESS
        else:
            crop_size = self.c_size
            i, j, h, w = self.crop_origin_y, self.crop_origin_x, crop_size, crop_size

        imgs = [F.crop(img, i, j, h, w) for img in imgs]
        if self.training:
            imgs = [F.resize(img, size=self.c_size) for img in imgs]

        if len(keypoints) > 0:
            keypoints = keypoints - [j, i]
            idx_mask = (keypoints[:, 0] >= 0) * (keypoints[:, 0] <= w) * (keypoints[:, 1] >= 0) * (keypoints[:, 1] <= h)
            keypoints = keypoints[idx_mask]
            ratio = self.c_size / crop_size
            keypoints = np.array([[k_x * ratio, k_y * ratio] for k_x, k_y in keypoints])
        else:
            keypoints = np.empty([0, 2])

        h, w = imgs[0].size
        return h, w, imgs, keypoints

    def augment(self, imgs, keypoints, gt_discrete):
        h, w = imgs[0].size

        # RANDOM FLIP
        if random.random() > 0.5:
            imgs = [F.hflip(img) for img in imgs]
            gt_discrete = np.fliplr(gt_discrete)
            if len(keypoints) > 0:
                keypoints[:, 0] = w - keypoints[:, 0]
        gt_discrete = np.expand_dims(gt_discrete, 0)

        return imgs, keypoints, gt_discrete

    def load_keypoints(self, img_path):
        """
        load ground truth from the JSON file and put them into an appropriate format
        :param img_path: path to the last image in the sequence
        :return:
        """
        gt_path = img_path.replace(".jpg", ".json")
        with open(gt_path) as json_file:
            json_data = json.load(json_file)
            regions = json_data[list(json_data)[0]]['regions']

        points = []
        for region in regions:
            shp_attr = region['shape_attributes']
            x = shp_attr['x']
            y = shp_attr['y']
            w = shp_attr['width']
            h = shp_attr['height']
            points.append((x + w/2.0, y + h/2.0))
        return np.array(points)

    def get_unnormed_item(self, idx):
        images = [Image.open(img_path).convert('RGB') for img_path in self.item_id_dict[idx]]
        if self.training:
            return None
        else:
            i, j, h, w = self.crop_origin_y, self.crop_origin_x, self.c_size, self.c_size
            imgs = [F.crop(img, i, j, h, w) for img in images]
            return imgs[-1]
