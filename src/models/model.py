import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models import resnet18
from collections import OrderedDict
from .encoder import Encoder
from .ConvRNN import CLSTM_cell


encoder_params = [
    [
        # (conv1_leaky_1): Conv2d(512, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (leaky_conv1_leaky_1): LeakyReLU(negative_slope=0.2, inplace=True)
        OrderedDict({'conv1_leaky_1': [512, 64, 3, 1, 1]}),
        #OrderedDict({'conv2_leaky_1': [64, 64, 3, 2, 1]}),
        #OrderedDict({'conv3_leaky_1': [96, 96, 3, 2, 1]}),
    ],
    [
        CLSTM_cell(shape=(16, 16), input_channels=64, filter_size=5, num_features=512),
        #CLSTM_cell(shape=(32,32), input_channels=64, filter_size=5, num_features=96),
        #CLSTM_cell(shape=(16,16), input_channels=96, filter_size=5, num_features=96)
    ]
]



class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.channels = 3

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

        #self.decoder = Decoder(decoder_params[0], decoder_params[1]).cuda()
        self.encoder = Encoder(encoder_params[0], encoder_params[1]).cuda()

        # endregion

        self.reg_layer = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.density_layer = nn.Sequential(nn.Conv2d(128, 1, 1), nn.ReLU())

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

        ((x, _), ) = self.encoder(x)        # ENCODER RETURNS ((hy, cy)), WE NEED hy

        print(x.shape)

        x = F.interpolate(x, scale_factor=16, mode='bilinear', align_corners=True)

        x = self.reg_layer(x)

        mu = self.density_layer(x)

        B, C, H, W = mu.size()
        mu_sum = mu.view([B, -1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        mu_normed = mu / (mu_sum + 1e-6)
        return mu, mu_normed
