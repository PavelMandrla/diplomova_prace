import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models import resnet18
from collections import OrderedDict

from .ConvRNN import CLSTM_cell

def make_layers(block):
    layers = []
    for layer_name, v in block.items():
        if 'pool' in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2])
            layers.append((layer_name, layer))
        elif 'deconv' in layer_name:
            transposeConv2d = nn.ConvTranspose2d(in_channels=v[0],
                                                 out_channels=v[1],
                                                 kernel_size=v[2],
                                                 stride=v[3],
                                                 padding=v[4])
            layers.append((layer_name, transposeConv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name, nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        elif 'conv' in layer_name:
            conv2d = nn.Conv2d(in_channels=v[0],
                               out_channels=v[1],
                               kernel_size=v[2],
                               stride=v[3],
                               padding=v[4])
            layers.append((layer_name, conv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name, nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        else:
            raise NotImplementedError
    return nn.Sequential(OrderedDict(layers))


class Decoder(nn.Module):
    def __init__(self, subnets, rnns):
        super().__init__()
        assert len(subnets) == len(rnns)

        self.blocks = len(subnets)

        for index, (params, rnn) in enumerate(zip(subnets, rnns)):
            setattr(self, 'rnn' + str(self.blocks - index), rnn)
            setattr(self, 'stage' + str(self.blocks - index), make_layers(params))

    def forward_by_stage(self, inputs, state, subnet, rnn):
        inputs, state_stage = rnn(inputs, state, seq_len=10)
        seq_number, batch_size, input_channel, height, width = inputs.size()
        inputs = torch.reshape(inputs, (-1, input_channel, height, width))
        inputs = subnet(inputs)
        inputs = torch.reshape(inputs, (seq_number, batch_size, inputs.size(1), inputs.size(2), inputs.size(3)))
        return inputs

        # input: 5D S*B*C*H*W

    def forward(self, hidden_states):
        print('hidden states len: ', len(hidden_states))
        inputs = self.forward_by_stage(None, hidden_states[-1], getattr(self, 'stage3'), getattr(self, 'rnn3'))
        for i in list(range(1, self.blocks))[::-1]:
            inputs = self.forward_by_stage(inputs, hidden_states[i - 1],
                                           getattr(self, 'stage' + str(i)),
                                           getattr(self, 'rnn' + str(i)))
        inputs = inputs.transpose(0, 1)  # to B,S,1,64,64
        return inputs


class Encoder(nn.Module):
    def __init__(self, subnets, rnns):
        super().__init__()
        assert len(subnets) == len(rnns)
        self.blocks = len(subnets)

        for index, (params, rnn) in enumerate(zip(subnets, rnns), 1):
            # index sign from 1
            setattr(self, 'stage' + str(index), make_layers(params))
            setattr(self, 'rnn' + str(index), rnn)

    def forward_by_stage(self, inputs, subnet, rnn):
        seq_number, batch_size, input_channel, height, width = inputs.size()
        inputs = torch.reshape(inputs, (-1, input_channel, height, width))
        inputs = subnet(inputs)
        print(inputs.size(1), inputs.size(2), inputs.size(3))
        inputs = torch.reshape(inputs, (seq_number, batch_size, inputs.size(1), inputs.size(2), inputs.size(3)))
        outputs_stage, state_stage = rnn(inputs, None)
        return outputs_stage, state_stage

    def forward(self, inputs):
        inputs = inputs.transpose(0, 1)  # to S,B,1,64,64
        hidden_states = []
        for i in range(1, self.blocks + 1):
            inputs, state_stage = self.forward_by_stage( inputs, getattr(self, 'stage' + str(i)), getattr(self, 'rnn' + str(i)))
            hidden_states.append(state_stage)
        return tuple(hidden_states)


encoder_params = [
    [
        # (conv1_leaky_1): Conv2d(512, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (leaky_conv1_leaky_1): LeakyReLU(negative_slope=0.2, inplace=True)
        OrderedDict({'conv1_leaky_1': [512, 64, 3, 1, 1]}),
        # OrderedDict({'conv2_leaky_1': [64, 64, 3, 2, 1]}),
        # OrderedDict({'conv3_leaky_1': [96, 96, 3, 2, 1]}),
    ],
    [
        CLSTM_cell(shape=(16, 16), input_channels=64, filter_size=5, num_features=96),
        # CLSTM_cell(shape=(32,32), input_channels=64, filter_size=5, num_features=96),
        # CLSTM_cell(shape=(16,16), input_channels=96, filter_size=5, num_features=96)
    ]
]

decoder_params = [
    [
        OrderedDict({'deconv1_leaky_1': [512, 512, 4, 2, 1]}),
        OrderedDict({'deconv2_leaky_1': [96, 96, 4, 2, 1]}),
        OrderedDict({
            'conv3_leaky_1': [64, 16, 3, 1, 1],
            'conv4_leaky_1': [16, 1, 1, 1, 0]
        }),
    ],
    [
        CLSTM_cell(shape=(16, 16), input_channels=512, filter_size=5, num_features=96),
        CLSTM_cell(shape=(32, 32), input_channels=96,  filter_size=5, num_features=96),
        CLSTM_cell(shape=(64, 64), input_channels=96,  filter_size=5, num_features=64),
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
        # in DM-Count
        #   features -> bilinear upsample -> reg_layer -> density layer
        # I want to do
        #   features -> ConvLSTM -> bilinear upsample -> reg_layer -> density layer

        # x = x.unsqueeze(dim=0)
        # print(x.shape)
        # self.ConvLSTM(x)

        x = self.features(x[0])
        x = x.unsqueeze(dim=1)
        print(x.shape)

        x = self.encoder(x)
        #x = self.decoder((x))

        x = F.interpolate(x, scale_factor=16, mode='bilinear', align_corners=True)

        x = self.reg_layer(x)

        mu = self.density_layer(x)

        B, C, H, W = mu.size()
        mu_sum = mu.view([B, -1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        mu_normed = mu / (mu_sum + 1e-6)
        return mu, mu_normed
