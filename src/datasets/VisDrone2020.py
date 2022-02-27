import os
import numpy as np
from glob import glob
from datasets import SequentialDataset


class VisDrone2020(SequentialDataset):

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

        dir_list_path = os.path.join(self.data_path, 'trainlist.txt' if self.training else 'testlist.txt')
        with open(dir_list_path) as dir_list_file:
            src_dirs = [line.rstrip() for line in dir_list_file]

        for dir_name in src_dirs:
            subdir = os.path.join(self.data_path, 'sequences/' + dir_name)
            items = glob(os.path.join(subdir, '*.jpg'))
            items.sort()

            sequence_gt = self.load_sequence_keypoints(dir_name)

            for i, item in enumerate(items):
                if not self.training and i < max_overall_length:
                    continue
                if i >= overall_length:
                    seq_start, seq_end = self.get_seq_bounds(i)
                    self.item_id_dict[top_i] = items[seq_start:seq_end:self.stride]
                    self.keypoints.append(np.array(sequence_gt[i]))
                    top_i += 1

    def load_sequence_keypoints(self, dir_name):
        result = []
        gt_path = os.path.join(self.data_path, 'annotations/' + dir_name + '.txt')
        with open(gt_path) as gt_file:
            for line in gt_file:
                data = line.split(',')
                i, x, y = int(data[0]), float(data[1]), float(data[2])
                if i > len(result):
                    result.append([])
                result[i - 1].append((x, y))
        return result








