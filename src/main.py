"""
import torch
import cv2
import numpy as np
from models.model import MyModel
from datasets.fdst import FDST
from torch.utils.data import DataLoader
from testing.utils import *
import matplotlib.pyplot as plt

#ssh -R port,kde běží jupyter notebook,  -

model_path = './save_dir/40_ckpt.tar'

dataset = FDST("../datasets/our_dataset", training=True, sequence_len=5)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MyModel(model_path)
model = model.eval().to(device)


#eval_video(model_path, '../datasets/VSB/20211005_120723.MOV')
range_real_time(model, dataset, device, 220, 700)
#animate_range(model, dataset, device)
"""


import torch
from torch.utils.data import DataLoader
from datasets.fdst import FDST
from testing.utils import show_image


dataset_path = '../datasets/our_dataset'
model_path = './save_dir/40_ckpt.tar'

dataset = FDST(dataset_path, training=False, sequence_len=5)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

a, b, c = next(iter(dataloader))
show_image(a, b[0])


