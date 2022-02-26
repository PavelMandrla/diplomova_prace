import os
import re
from .sequentialDataset import SequentialDataset
from glob import glob
import xml.etree.ElementTree as ET
import numpy as np


class PETS(SequentialDataset):

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
        super().__init__(
            root_path,
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

        subdirs = os.listdir(os.path.join(self.data_path))
        gt_regex = re.compile(".*xml")

        for subdir in subdirs:
            item_path = os.path.join(self.data_path, subdir)
            subdir_files = [str(x) for x in os.listdir(item_path)]
            gt_file = list(filter(gt_regex.match, subdir_files))[0]
            frame_gt = self.parse_gt(os.path.join(str(item_path), gt_file))

            items = glob(os.path.join(str(item_path), '*.jpg'))
            items.sort()

            for i, item in enumerate(items):
                if not self.training and i < max_overall_length:
                    continue
                if i >= overall_length:
                    seq_start, seq_end = self.get_seq_bounds(i)
                    self.item_id_dict[top_i] = items[seq_start:seq_end:self.stride]
                    self.keypoints.append(frame_gt[i])
                    top_i += 1

    def parse_gt(self, gt_file):
        frames = []

        tree = ET.parse(gt_file)
        root = tree.getroot()

        for frame in root.findall('frame'):
            people = []
            for person in frame.find('objectlist').findall('object'):
                box = person.find('box')
                x = float(box.get('xc'))
                y = float(box.get('yc'))
                w = float(box.get('w'))
                h = float(box.get('h'))
                people.append((x + w / 2.0, y + h / 2.0))

            frames.append(np.array(people))
        return frames
