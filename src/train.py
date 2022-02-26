import argparse
import os
import torch
from trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--device',                         default='0',    help='assign device')
    parser.add_argument('--lr',                 type=float, default=1e-5,   help='initial learning rate')
    parser.add_argument('--weight-decay',       type=float, default=1e-4,   help='weight decay')
    parser.add_argument('--max-epoch',          type=int,   default=40,     help='max training epoch')
    parser.add_argument('--crop-size',          type=int,   default=512,    help='crop size of the train image')
    parser.add_argument('--wot',                type=float, default=0.1,    help='weight on OT loss')
    parser.add_argument('--wtv',                type=float, default=0.01,   help='weight on TV loss')
    parser.add_argument('--reg',                type=float, default=10.0,   help='entropy regularization in sinkhorn')
    parser.add_argument('--num-of-iter-in-ot',  type=int,   default=100,    help='sinkhorn iterations')
    parser.add_argument('--norm-cood',          type=int,   default=0,      help='whether to norm cood when computing distance')

    parser.add_argument('--dataset_path',       type=str,   default='../datasets/our_dataset', help='paht to the dataset')
    parser.add_argument('--stride',             type=int,   default=1,      help='stride between dataset images in sequence')
    parser.add_argument('--sequence_length',    type=int,   default=5,      help='length of input sequence')
    parser.add_argument('--save_dir',           type=str,   default='../save_dir/m4', help='folder to which to save the trained models')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu
    trainer = Trainer(args)
    trainer.train()
