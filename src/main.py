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

dataset = FDST("../datasets/our_dataset", training=False, sequence_len=5)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MyModel(model_path)
model = model.eval().to(device)


eval_video(model_path, '../datasets/VSB/20211005_120723.MOV')
#range_real_time(model, dataset, device, 220, 700)
#animate_range(model, dataset, device)


