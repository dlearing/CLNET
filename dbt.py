import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict


class SemanticGroupingLayer(nn.Module):

    def __init__(self, in_channels, out_channels, batch_size, groups, device):
        super(SemanticGroupingLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.batch_size = batch_size
        self.groups = groups

        self.sg_conv = nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(in_channels, out_channels, 1, bias=False)),
            ("bn", nn.BatchNorm2d(out_channels)),
            ("relu", nn.ReLU(inplace=True))
        ]))

        self.gt = torch.ones(self.groups).diag().to(device)
        self.gt = self.gt.reshape((1, 1, self.groups, self.groups))
        # 1 1 groups groups
        self.gt = self.gt.repeat((1, int((self.out_channels / self.groups) ** 2), 1, 1))
        # 1 out_channels/groups groups groups
        self.gt = F.pixel_shuffle(self.gt, upscale_factor=int(self.out_channels / self.groups))
        # 1 1 out_channels out_channels
        self.gt = self.gt.reshape((1, self.out_channels ** 2))

        self.loss = 0

    def forward(self, x):

        # calculate act
        act = self.sg_conv(x)

        b, c, w, h = x.shape

        # loss
        print('~~~~~~~~~~~~~~~~~~~~~~~')
        print(act.shape)
        print(self.in_channels)
        print(self.out_channels)
        tmp = act + 1e-3

        tmp = tmp.reshape((-1, w*h))
        #tmp = F.instance_norm(tmp)
        tmp = tmp.reshape((-1, self.out_channels, w*h))

        tmp = tmp.permute(1, 0, 2).reshape(self.out_channels, -1)

        print(torch.matmul(tmp, tmp.t()).shape)

        co_matrix = torch.matmul(tmp, tmp.t()).reshape((1, self.out_channels**2))

        co_matrix /= self.batch_size

        loss = torch.sum((co_matrix-self.gt)*(co_matrix-self.gt)*0.001, dim=1).repeat(self.batch_size)

        self.loss = loss/((self.out_channels/512.0)**2)

        return act


class PositionEncoding(object):
    def __init__(self, group, device):
        self.group = group
        self.device = device

    def cal_angle(self, position, hid_idx):
        return position * 1.0 / np.power(1.5, 2.0*hid_idx/self.group)

    def get_position_angle_vec(self, position):
        return [self.cal_angle(position, hid_j) for hid_j in range(self.group)]

    def __call__(self, bias_shape):

        n_position = int(bias_shape/self.group)

        sinusoid_table = np.array([self.get_position_angle_vec(pos_i) for pos_i in range(n_position)])

        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        bias = torch.from_numpy(np.array(sinusoid_table.flat)).to(dtype=torch.float)
        return bias


class GroupBilinearLayer(nn.Module):

    def __init__(self, groups, channels, device):
        super(GroupBilinearLayer, self).__init__()

        self.groups = groups
        self.channels = channels
        self.channels_per_group = int(self.channels/self.groups)

        self.fc = nn.Linear(channels, channels)
        self.bn = nn.BatchNorm2d(channels)

        self.position_encoding = PositionEncoding(groups, device)
        self.fc.bias.data = self.position_encoding(self.fc.bias.data.shape[0])

    def forward(self, x):

        b, c, w, h = x.shape
        print('******************8888')
        print(x.shape)
        tmp = x.permute(0, 2, 3, 1).reshape((-1, self.channels))
        print('******************999')
        print(tmp.shape)
        tmp += self.fc(tmp)
        print('******************101010')
        print(tmp.shape)
        tmp = tmp.reshape(-1, self.groups, self.channels_per_group)
        print('******************111111')
        print(tmp.shape)

        tmp_t = tmp.permute(0, 2, 1)
        print('******************12121221121212')
        print(torch.bmm(tmp_t, tmp).shape)
        tmp = torch.tanh(torch.bmm(tmp_t, tmp)/32).reshape(-1, w, h, self.channels_per_group**2)

        print('******************13131313131')
        print(tmp.shape)
        tmp = F.interpolate(tmp, (h, c), mode="bilinear")
        tmp = tmp.permute(0, 3, 1, 2)

        out = x + self.bn(tmp)
        return out


class DBTNetBlock(nn.Module):

    def __init__(self, inplanes, planes, batch_size, device, groups=None, downsample=False, use_dbt=False):
        super(DBTNetBlock, self).__init__()

        self.block = nn.Sequential()
        expansion = 4
        if downsample:
            stride = 2
        else:
            stride = 1

        if use_dbt:
            self.sg_layer = SemanticGroupingLayer(inplanes, planes, batch_size, groups, device)
            self.gb_layer = GroupBilinearLayer(groups, planes, device)
            self.block.add_module("sg_layer", self.sg_layer)
            self.block.add_module("gb_layer", self.gb_layer)
        else:
            self.block.add_module("conv_1", nn.Conv2d(inplanes, planes, 1, bias=False))
            self.block.add_module("bn_1", nn.BatchNorm2d(planes))
            self.block.add_module("relu_1", nn.ReLU(inplace=True))

        self.block.add_module("conv 2", nn.Conv2d(planes, planes, 3, stride, 1, bias=False))
        self.block.add_module("bn_2", nn.BatchNorm2d(planes))
        self.block.add_module("relu_2", nn.ReLU(inplace=True))

        self.block.add_module("conv_3", nn.Conv2d(planes, planes*expansion, 1, bias=False))

        self.downsample = nn.Sequential()
        if downsample:
            self.downsample.add_module("conv_downsample", nn.Conv2d(inplanes, planes*expansion,
                                                                    1, 2, bias=False))
            self.downsample.add_module("bn_downsample", nn.BatchNorm2d(planes*expansion))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.downsample(x)
        x = self.block(x)

        output = self.relu(x+identity)
        return output


class DBTNet(nn.Module):

    def __init__(self, layers, batch_size, device, classes=1000):
        super(DBTNet, self).__init__()
        self.inplanes = 64
        self.features = nn.Sequential()
        self.sg_layers = list()
        self.expansion = 4

        # -------------
        # input conv
        # -------------
        # BN-Conv-BN-ReLU-Max pooling
        self.input_ = nn.Sequential(OrderedDict([
            ("input_conv", nn.Conv2d(3, self.inplanes, 7, 2, 3, bias=False)),
            ("input_bn", nn.BatchNorm2d(self.inplanes)),
            ("input_activate", nn.ReLU(inplace=True)),
            ("input_maxpooling", nn.MaxPool2d(3, 2, 1))
        ]))
        self.features.add_module("input_features", self.input_)

        # -------------
        # resnet-dbt block
        # -------------
        self.features.add_module("block_1", self._make_layer("block_1", 64, layers[0],
                                                             batch_size, device, use_dbt=False))
        self.features.add_module("block_2", self._make_layer("block_2", 128, layers[1],
                                                             batch_size, device, use_dbt=False))
        self.features.add_module("block_3", self._make_layer("block_3", 256, layers[2],
                                                             batch_size, device, groups=16, use_dbt=True))
        self.features.add_module("block_4", self._make_layer("block_4", 512, layers[3],
                                                             batch_size, device, groups=16, use_dbt=True))

        # -------------
        # last DBT module
        # ------------
        self.last_sg_layer = SemanticGroupingLayer(self.inplanes, self.inplanes, batch_size, 32, device)
        self.last_gb_layer = GroupBilinearLayer(32, self.inplanes, device)
        self.features.add_module("last_sg_layer", self.last_sg_layer)
        self.features.add_module("last_gb_layer", self.last_gb_layer)
        self.sg_layers.append(self.last_sg_layer)

        # -------------
        # output
        # -------------
        # channels: inplanes -> classes
        self.features.add_module("output_11conv", nn.Conv2d(self.inplanes, classes, 1))
        self.avgpooling = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, basic_name, base_planes, blocks_num, batch_size, device, groups=None, use_dbt=False):
        layers = nn.Sequential()

        dbt_block = DBTNetBlock(inplanes=self.inplanes, planes=base_planes, batch_size=batch_size,
                                device=device, groups=groups, downsample=True, use_dbt=use_dbt)
        layers.add_module("{}_dbt_1".format(basic_name), dbt_block)
        if use_dbt:
            self.sg_layers.append(dbt_block.sg_layer)

        self.inplanes = base_planes * self.expansion

        for i in range(blocks_num-1):
            dbt_block = DBTNetBlock(inplanes=self.inplanes, planes=base_planes, batch_size=batch_size,
                                    device=device, groups=groups, use_dbt=use_dbt)

            layers.add_module("{}_dbt_{}".format(basic_name, i+2), dbt_block)

            if use_dbt:
                self.sg_layers.append(dbt_block.sg_layer)

        return layers

    def forward(self, x):
        x = self.features(x)
        x = self.avgpooling(x)
        x = x.view(x.shape[0], -1)

        # loss
        dbt_loss = None
        sg_layer_count = 0
        for sg_layer in self.sg_layers:
            if sg_layer_count == 0:
                dbt_loss = sg_layer.loss
            else:
                dbt_loss += sg_layer.loss

            sg_layer_count += 1

        dbt_loss /= sg_layer_count
        return x, dbt_loss


dbt_spec = {"DBTNet-50": [3, 4, 6, 3],
            "DBTNet-101": [3, 4, 23, 3]}


def dbt(model_name, batch_size, device):

    assert model_name in dbt_spec, "the model name: {} is not ."

    model = DBTNet(dbt_spec[model_name], batch_size, device).to(device)
    return model


if __name__ == "__main__":

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    dbt_net = dbt("DBTNet-50", 2, device)

    #dbt_net = DBTNet([3, 4, 23, 3], 48, device).to(device)
    # dbt_net = DBTNet([3, 4, 6, 3], 48, device).to(device)
    #tensor = Variable(torch.unsqueeze(np.tensor(np.ones()), dim=0).float(), requires_grad=False)
    #a = torch.Tensor(np.ones((1,3, 256, 256)))
    test_noise = torch.randn((2,3, 1024, 1024)).to(device)
    #dbt_net.eval('(1,3, 256, 256]')
    output, loss = dbt_net(test_noise)
    #dbt_net.eval()
    print(output.shape, loss.shape)