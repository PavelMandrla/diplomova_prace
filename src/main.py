from models.model import *
from datasets import *
from torch.utils.data import DataLoader
from testing.utils import *


def initialize(dataset_name, model_path):
    if dataset_name == 'FDST':
        dataset_path = '../datasets/our_dataset'
        input_size = (1920, 1080)
    elif dataset_name == 'PETS':
        dataset_path = '../datasets/PETS'
        input_size = (768, 576)
    else: # == 'VisDrone':
        dataset_path = '../datasets/VisDrone2020-CC'
        input_size = (1920, 1080)

    # model = MyModelAlternative(model_path, input_size=input_size)
    model = MyModel(model_path, input_size=input_size)

    if dataset_name == 'FDST':
        dataset = FDST(
            dataset_path,
            training=False,
            sequence_len=model.seq_len,
            crop_size=input_size,
            stride=model.stride,
            max_sequence_len=5,
            max_stride=5)
    elif dataset_name == 'PETS':
        dataset = PETS(
            dataset_path,
            training=False,
            sequence_len=model.seq_len,
            crop_size=input_size,
            stride=model.stride,
            max_sequence_len=7,
            max_stride=5)
    else: # == 'VisDrone':
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

    return model, dataloader, dataset, device


# dataset_path = '../datasets/UCF-QNRF/UCF-QNRF_ECCV18'
# input_size = (1920, 1080)
#
# model = MyModel(input_size=input_size)
# dataset = QNRF(dataset_path, training=False, crop_size=input_size)
#
# dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = model.eval().to(device)

dataset_name = 'FDST'
# model_path = './trained_models/len5_stride3.tar'
model_path = '../save_dir/3_3/15_ckpt.tar'
model, dataloader, dataset, device = initialize(dataset_name, model_path)

evaluate_dataset(model, dataloader, device, 'old_len3_stride_3.csv')

# plot_timeseries(model, dataset, device, 0, 100)
# eval_video(model, '../datasets/VSB/20211005_120808.MOV', device, stride=3, sequence_len=5)
# range_real_time(model, dataset, device, 0, 700)
# animate_range(model, dataset, device)
# animate_video(model, device, '../datasets/VSB/20211005_120808.MOV', './save_dir/entry.mp4')




