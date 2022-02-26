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
from .sequentialDataset import SequentialDataset


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


class FDST(SequentialDataset):

    def __init__(
            self,
            root_path,
            training=True,
            sequence_len=5,
            crop_size=(512, 512),
            crop_origin=(0, 0),
            downsample_ratio=2,
            stride=1,
            max_sequence_len=0,
            max_stride=0
    ):
        """
        :param root_path: path to the root directory of the dataset
        :param training: indicates, whether we want to load the data from the train_data or their test_data subdirectory
        :param sequence_len: number of images, that are contained in one item
        :param crop_size: the size of the resulting images (width, height)
        :param crop_origin: if the crop is not random (in trainig), sets the position, from which to crop the image
        :param downsample_ratio: downsample_ratio
        :param stride: stride in the sequence of images
        :param max_sequence_len: for comparison between dataset -> ensures, that the same inputs are compared when using different stride and seq_
        :param max_stride: for comparison between dataset -> ensures, that the same inputs are compared when using different stride and seq_
        """

        data_path = os.path.join(root_path, "train_data" if training else "test_data")
        super().__init__(
            data_path,
            training,
            sequence_len,
            crop_size,
            crop_origin,
            downsample_ratio,
            stride,
            max_sequence_len,
            max_stride
        )

    def index_inputs(self):
        top_i = 0
        overall_length = self.seq_len * self.stride
        max_overall_length = self.max_seq_len * self.max_stride

        dir_items = os.listdir(self.data_path)      # GET ALL ITEMS IN DIRECTORY
        for name in dir_items:
            item_path = os.path.join(self.data_path, name)
            if os.path.isdir(item_path):                        # IF THE ITEM IS A DIRECTORY
                items = glob(os.path.join(item_path, '*.jpg'))
                items.sort()

                for i, item in enumerate(items):
                    if not self.training and i < max_overall_length:
                        continue
                    if i >= overall_length:
                        seq_start, seq_end = self.get_seq_bounds(i)
                        self.item_id_dict[top_i] = items[seq_start:seq_end:self.stride]
                        self.keypoints.append(self.load_keypoints(item))
                        top_i += 1

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
