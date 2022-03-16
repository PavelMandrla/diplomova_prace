import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models import resnet18
from collections import OrderedDict
from .encoder import Encoder
from .ConvRNN import CLSTM_cell


class MyModel(nn.Module):
    def __init__(self, model_path=None, input_size=(512, 512), sequence_len=5, stride=1):
        """
        :param model_path: If pretrained, insert the path to the file
        :param input_size: Size of the input images (width, height)
        :param sequence_len: Length of the input sequence
        :param stride: Not really used for anything in the model, but it is good to know, when creating a dataloader
        """
        super(MyModel, self).__init__()
        self.channels = 3
        self.input_size = input_size
        self.seq_len = sequence_len
        self.stride = stride

        if model_path is not None:
            saved_model = torch.load(model_path)
            self.seq_len = saved_model['sequence_length']
            self.stride = saved_model['stride']

        # region INIT FEATURES
        resnet = resnet18(True, True)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        # endregion

        # region INIT CONVLSTM
        features_size = self.get_size_after_features(input_size)
        self.encoder = Encoder(
            [
                OrderedDict({'conv1_leaky_1': [512, 64, 3, 1, 1]}),
                OrderedDict({'conv2_leaky_1': [64, 64, 3, 1, 1]}),
            ],
            [
                CLSTM_cell(
                    shape=(features_size[1], features_size[0]),
                    input_channels=64,
                    filter_size=5,
                    num_features=64,
                    seq_len=self.seq_len),
                CLSTM_cell(
                    shape=(features_size[1], features_size[0]),
                    input_channels=64,
                    filter_size=5,
                    num_features=512,
                    seq_len=self.seq_len)
            ])
        # endregion

        self.reg_layer = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.density_layer = nn.Sequential(nn.Conv2d(128, 1, 1), nn.ReLU())

        if model_path is not None:
            saved_model = torch.load(model_path)
            self.load_state_dict(saved_model['model_state_dict'])

    def features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def forward(self, x):
        x = self.features(x[0])
        x = x.unsqueeze(dim=0)

        x, _ = self.encoder(x)[-1]  # ENCODER RETURNS ((hy, cy)), WE NEED hy

        x = F.interpolate(x, scale_factor=16, mode='bilinear', align_corners=True)

        x = self.reg_layer(x)

        mu = self.density_layer(x)

        B, C, H, W = mu.size()
        mu_sum = mu.view([B, -1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        mu_normed = mu / (mu_sum + 1e-6)
        return mu, mu_normed

    def get_size_after_features(self, input_size):
        input = torch.zeros(1,3,input_size[1], input_size[0])
        output = self.features(input)
        return output.shape[:-3:-1]  # return last two in reverse order (width, height)

    def save(self, path):
        torch.save({
            'sequence_length': self.seq_len,
            'stride': self.stride,
            'model_state_dict': self.state_dict()
        }, path)


class MyModel_(nn.Module):
    def __init__(self, model_path=None, input_size=(512, 512), sequence_len=5, stride=1):
        """
        :param model_path: If pretrained, insert the path to the file
        :param input_size: Size of the input images (width, height)
        :param sequence_len: Length of the input sequence
        :param stride: Not really used for anything in the model, but it is good to know, when creating a dataloader
        """
        super(MyModel, self).__init__()
        self.channels = 3
        self.input_size = input_size
        self.seq_len = sequence_len
        self.stride = stride

        if model_path is not None:
            saved_model = torch.load(model_path)
            self.seq_len = saved_model['sequence_length']
            self.stride = saved_model['stride']

        # region INIT FEATURES
        resnet = resnet18(True, True)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        # endregion

        # region INIT CONVLSTM
        features_size = self.get_size_after_features(input_size)
        self.encoder = Encoder(
            [
                # OrderedDict({'conv1_leaky_1': [512, 64, 3, 1, 1]}),
                OrderedDict(),
                # OrderedDict({'conv2_leaky_1': [64, 64, 3, 1, 1]}),
                OrderedDict(),
                OrderedDict(),
            ],
            [
                CLSTM_cell(
                    shape=(features_size[1], features_size[0]),
                    input_channels=512,
                    filter_size=5,
                    num_features=512,
                    seq_len=self.seq_len),
                CLSTM_cell(
                    shape=(features_size[1], features_size[0]),
                    input_channels=512,
                    filter_size=5,
                    num_features=256,
                    seq_len=self.seq_len),
                CLSTM_cell(
                    shape=(features_size[1], features_size[0]),
                    input_channels=256,
                    filter_size=5,
                    num_features=256,
                    seq_len=self.seq_len)
            ])
        # endregion

        self.reg_layer = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        )
        self.density_layer = nn.Sequential(nn.Conv2d(128, 1, 1), nn.ReLU())

        if model_path is not None:
            saved_model = torch.load(model_path)
            self.load_state_dict(saved_model['model_state_dict'])

    def features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def forward(self, x):
        x = self.features(x[0])
        x = x.unsqueeze(dim=0)

        x, _ = self.encoder(x)[-1]  # ENCODER RETURNS ((hy, cy)), WE NEED hy

        x = F.interpolate(x, scale_factor=16, mode='bilinear', align_corners=True)

        x = self.reg_layer(x)

        mu = self.density_layer(x)

        B, C, H, W = mu.size()
        mu_sum = mu.view([B, -1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        mu_normed = mu / (mu_sum + 1e-6)
        return mu, mu_normed

    def get_size_after_features(self, input_size):
        input = torch.zeros(1,3,input_size[1], input_size[0])
        output = self.features(input)
        return output.shape[:-3:-1]  # return last two in reverse order (width, height)

    def save(self, path):
        torch.save({
            'sequence_length': self.seq_len,
            'stride': self.stride,
            'model_state_dict': self.state_dict()
        }, path)
