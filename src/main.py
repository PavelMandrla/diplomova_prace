from models.model import MyModel
from datasets import FDST, PETS, VisDrone2020
from torch.utils.data import DataLoader
from testing.utils import *


#dataset_path = '../datasets/our_dataset'
#input_size = (1920, 1080)

# dataset_path = '../datasets/PETS'
# input_size = (768, 576)

dataset_path = '../datasets/VisDrone2020-CC'
input_size = (1920, 1080)

model_path = './trained_models/len5_stride3.tar'
model = MyModel(model_path, input_size=input_size)

# dataset = FDST(
#     dataset_path,
#     training=False,
#     sequence_len=model.seq_len,
#     crop_size=input_size,
#     stride=model.stride,
#     max_sequence_len=7,
#     max_stride=5)

# dataset = PETS(
#     dataset_path,
#     training=False,
#     sequence_len=model.seq_len,
#     crop_size=input_size,
#     stride=model.stride,
#     max_sequence_len=7,
#     max_stride=5)

dataset = VisDrone2020(
    dataset_path,
    training=False,
    sequence_len=model.seq_len,
    crop_size=input_size,
    stride=model.stride,
    max_sequence_len=3,
    max_stride=5)

dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.eval().to(device)

#print(model)

# evaluate_dataset(model, dataloader, device, 'pets.csv')

# plot_timeseries(model, dataset, device, 0, 100)
# eval_video(model, '../datasets/VSB/20211005_120808.MOV', device, stride=3, sequence_len=5)
range_real_time(model, dataset, device, 0, 700)
# animate_range(model, dataset, device)
# animate_video(model, device, '../datasets/VSB/20211005_120808.MOV', './save_dir/entry.mp4')




