import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn import init
from torch.nn.modules.utils import _pair

from utils.geometry_helper import unfold_padding, get_weight_alpha


class HexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, level, stride=1, bias=False):
        """ Hexagonal convolution proposed in [1].

        References
        ----------
        [1] orientation-aware semantic segmentation on icosahedron spheres, ICCV2019

        """
        super(HexConv2d, self).__init__()
        if stride != 1 and stride != 2:
            raise ValueError("stride must be 1 or 2")
        outlevel = level if stride == 1 else level - 1

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.stride = _pair(stride)
        # weight
        weight_trainable = torch.Tensor(out_channels, in_channels, 7)
        nn.init.kaiming_uniform_(weight_trainable, mode='fan_in', nonlinearity='relu')
        self.weight = Parameter(weight_trainable)  # adding zero

        #         self._w1_index = [
        #             [-1, 5, 4],
        #             [0, 6, 3],
        #             [1, 2, -1]
        #         ]
        #         self._w2_index = [
        #             [-1, 0, 5],
        #             [1, 6, 4],
        #             [2, 3, -1]
        #         ]

        _w1_index = [
            5, 4,
            0, 6, 3,
            1, 2
        ]
        _w2_index = [
            0, 5,
            1, 6, 4,
            2, 3
        ]

        self.register_buffer('w1_index', torch.tensor(_w1_index))
        self.register_buffer('w2_index', torch.tensor(_w2_index))

        alpha = torch.from_numpy(get_weight_alpha(outlevel)).float()
        self.register_buffer('alpha', alpha)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
            fan_in = in_channels * 9
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size=(hexagonal)'
             ', stride={stride}')
        return s.format(**self.__dict__)

    def get_hex_weight(self):
        weight1 = F.pad(torch.index_select(self.weight, -1, self.w1_index), (1, 1))
        weight2 = F.pad(torch.index_select(self.weight, -1, self.w2_index), (1, 1))
        out_ch, in_ch = weight1.shape[:2]
        return weight1.view(out_ch, in_ch, 3, 3), weight2.view(out_ch, in_ch, 3, 3)

    def forward(self, input):
        x = unfold_padding(input)
        weight1, weight2 = self.get_hex_weight()

        outputs = [None for _ in range(5)]
        for i in range(5):
            feat1 = F.conv2d(x[i], weight1, self.bias, self.stride)
            feat2 = F.conv2d(x[i], weight2, self.bias, self.stride)
            outputs[i] = self.alpha * feat1 + (1 - self.alpha) * feat2
        return outputs


class UnfoldReLU(nn.ReLU):
    def forward(self, x):
        out = [super(UnfoldReLU, self).forward(x[i]) for i in range(5)]
        return out


# class UnfoldBatchNorm2d(nn.BatchNorm2d):
#     def forward(self, x):
#         b, c, h, w = x[0].shape
#         # batch => len(x)*batch
#         out_cat = super(UnfoldBatchNorm2d, self).forward(torch.cat(x, dim=0))
#         out = [out_cat[b * i:b * (i + 1)] for i in range(5)]  # => list of b x c x h x w
#         return out
class UnfoldBatchNorm2d(nn.BatchNorm3d):
    def forward(self, x):
        stack_x = torch.stack(x, dim=2)
        stack_out = super(UnfoldBatchNorm2d, self).forward(stack_x)
        out = [stack_out[:, :, i] for i in range(5)]
        return out


class UnfoldConv2d(nn.Conv2d):
    def forward(self, x):
        out = [super(UnfoldConv2d, self).forward(x[i]) for i in range(5)]
        return out


class UnfoldMaxPool2d(nn.MaxPool2d):
    def forward(self, x):
        out = [super(UnfoldMaxPool2d, self).forward(x[i]) for i in range(5)]
        return out


class UnfoldUpsample(nn.Module):
    def __init__(self):
        super(UnfoldUpsample, self).__init__()
        self.up = partial(F.interpolate, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = unfold_padding(x, only_NE=True)
        h, w = x[0].shape[-2:]

        for i in range(5):
            x[i] = self.up(x[i], (2 * h - 1, 2 * w - 1))[..., 1:, :-1]

        return x
