import torch
import cv2
import numpy as np
from models.model import MyModel
from datasets.fdst import FDST
from torch.utils.data import DataLoader
from testing.utils import *
import matplotlib.pyplot as plt

#ssh -R port,kde běží jupyter notebook,  -

#model_path = './save_dir/40_ckpt.tar'
model_path = './trained_models/len5_stride3.tar'

dataset = FDST("../datasets/our_dataset", training=False, sequence_len=5, crop_size=(1280, 720), stride=3)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#model = MyModel(model_path, input_size=(1920, 1080))
model = MyModel(model_path, input_size=(1280, 720))
model = model.eval().to(device)


#plot_timeseries(model, dataset, device, 0, 100)
eval_video(model, '../datasets/VSB/20211005_120808.MOV', device, stride=3, sequence_len=5)
#range_real_time(model, dataset, device, 220, 700)
#animate_range(model, dataset, device)


