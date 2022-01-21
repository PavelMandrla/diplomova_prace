from torch.utils.data import Dataset
import random
import numpy as np
import torchvision.transforms.functional as F


class SequentialDataset(Dataset):

    def crop(self, imgs, keypoints):
        img_w, img_h = imgs[0].size
        st_size = 1.0 * min(img_w, img_h)
        assert st_size >= self.crop_size

        if self.training:
            # RANDOMLY CROP AT RANDOM SCALE
            crop_size = int(np.random.uniform(self.crop_size / 4, min(img_w, img_h)))  # RANDOMLY RESIZE THE IMAGE
            i = random.randint(0, img_h - crop_size)
            j = random.randint(0, img_w - crop_size)
            i, j, h, w = i, j, crop_size, crop_size     # TODO -> REFACTOR THIS MESS
        else:
            # TAKE REGULAR CUTOUT
            crop_size = self.crop_size
            i, j, h, w = self.crop_origin_y, self.crop_origin_x, crop_size, crop_size

        imgs = [F.crop(img, i, j, h, w) for img in imgs]
        if self.training:
            imgs = [F.resize(img, size=[self.crop_size]) for img in imgs]

        if len(keypoints) > 0:
            keypoints = keypoints - [j, i]
            idx_mask = (keypoints[:, 0] >= 0) * (keypoints[:, 0] <= w) * (keypoints[:, 1] >= 0) * (keypoints[:, 1] <= h)
            keypoints = keypoints[idx_mask]
            ratio = self.crop_size / crop_size
            keypoints = np.array([[k_x * ratio, k_y * ratio] for k_x, k_y in keypoints])
        else:
            keypoints = np.empty([0, 2])

        h, w = imgs[0].size
        return h, w, imgs, keypoints

    def __init__(self, training=True, crop_size=512, crop_origin_x=0, crop_origin_y=0):
        self.training = training
        self.crop_size = crop_size
        self.crop_origin_x = crop_origin_x
        self.crop_origin_y = crop_origin_y