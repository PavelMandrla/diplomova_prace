# This file contains the dataloader for the Fudan-ShanghaiTech dataset
# https://github.com/sweetyy83/Lstn_fdst_dataset

import os
import cv2
import torch
import random
import json
from glob import glob
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torchvision.transforms.functional as F
import numpy as np


def random_crop(im_h, im_w, crop_h, crop_w):
    res_h = im_h - crop_h
    res_w = im_w - crop_w
    i = random.randint(0, res_h)
    j = random.randint(0, res_w)
    return i, j, crop_h, crop_w


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
    discrete_map = torch.zeros(im_width * im_height).scatter_add_(0, index=p_index,
                                                                  src=torch.ones(im_width * im_height)).view(im_height,
                                                                                                             im_width).numpy()

    ''' slow method
    for p in points:
        p = np.round(p).astype(int)
        p[0], p[1] = min(h - 1, p[1]), min(w - 1, p[0])
        discrete_map[p[0], p[1]] += 1
    '''
    assert np.sum(discrete_map) == num_gt
    return discrete_map


class FDST(Dataset):

    def __init__(self, root_path, training=True, sequence_len=5, crop_size=512, downsample_ratio=2):
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
        self.d_ratio = downsample_ratio
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.item_id_dict = {}

        # get files from subdirecories and assign indicies to them
        top_i = 0
        subdirs = [os.path.join(self.data_path, name) for name in os.listdir(self.data_path) if os.path.isdir(os.path.join(self.data_path, name))]
        for subdir in subdirs:
            items = glob(os.path.join(subdir, '*.jpg'))
            items.sort()
            for i, item in enumerate(items):
                if i >= self.sequence_len:
                    self.item_id_dict[top_i] = items[i-self.sequence_len:i]
                    top_i += 1

    def __len__(self):
        return len(self.item_id_dict.keys())

    def __getitem__(self, idx):
        images = [Image.open(img_path).convert('RGB') for img_path in self.item_id_dict[idx]]
        keypoints = self.load_keypoints(self.item_id_dict[idx][-1])

        '''
        tensors = []
        for img_path in self.item_id_dict[idx]:
            img = cv2.imread(img_path)
            img_tensor = torch.from_numpy(img)
            img_tensor = img_tensor.permute(2, 0, 1)
            tensors.append(img_tensor)
        return torch.stack(tensors)
        '''
        return self.train_transform(images, keypoints)

    def train_transform(self, imgs, keypoints):
        wd, ht = imgs[0].size
        st_size = 1.0 * min(wd, ht)
        assert st_size >= self.c_size
        assert len(keypoints) >= 0
        i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
        imgs = [F.crop(img, i, j, h, w) for img in imgs]
        if len(keypoints) > 0:
            keypoints = keypoints - [j, i]
            idx_mask = (keypoints[:, 0] >= 0) * (keypoints[:, 0] <= w) * \
                       (keypoints[:, 1] >= 0) * (keypoints[:, 1] <= h)
            keypoints = keypoints[idx_mask]
        else:
            keypoints = np.empty([0, 2])

        gt_discrete = gen_discrete_map(h, w, keypoints)
        down_w = w // self.d_ratio
        down_h = h // self.d_ratio
        gt_discrete = gt_discrete.reshape([down_h, self.d_ratio, down_w, self.d_ratio]).sum(axis=(1, 3))
        assert np.sum(gt_discrete) == len(keypoints)

        if len(keypoints) > 0:
            if random.random() > 0.5:
                imgs = [F.hflip(img) for img in imgs]
                gt_discrete = np.fliplr(gt_discrete)
                keypoints[:, 0] = w - keypoints[:, 0]
        else:
            if random.random() > 0.5:
                imgs = [F.hflip(img) for img in imgs]
                gt_discrete = np.fliplr(gt_discrete)
        gt_discrete = np.expand_dims(gt_discrete, 0)

        imgs_transformed = [self.trans(img) for img in imgs]
        imgs_tensor = torch.stack(imgs_transformed)

        return imgs_tensor, torch.from_numpy(keypoints.copy()).float(), torch.from_numpy(gt_discrete.copy()).float()

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


