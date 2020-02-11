import torch
import torch.nn as nn

from models.unfold_nn import HexConv2d, UnfoldReLU, UnfoldBatchNorm2d, UnfoldMaxPool2d, UnfoldConv2d
from models.unfold_nn import UnfoldUpsample, UnfoldAvgPool2d

# ALPHA = None  # default alpha
ALPHA = 0.25  # default alpha


class OctHexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, level, stride=1,
                 alpha_in=None, alpha_out=None, type='normal', bias=False):
        super(OctHexConv2d, self).__init__()

        if alpha_in is None and ALPHA is not None:
            alpha_in = ALPHA
        if alpha_out is None and ALPHA is not None:
            alpha_out = ALPHA

        self.stride = stride
        self.type = type
        hf_ch_in = int(in_channels * (1 - alpha_in))
        hf_ch_out = int(out_channels * (1 - alpha_out))
        lf_ch_in = in_channels - hf_ch_in
        lf_ch_out = out_channels - hf_ch_out

        if type == 'first':
            self.hf_level = level if stride == 1 else level - 1
            self.lf_level = self.hf_level - 1
            if stride == 2:
                self.downsample = UnfoldAvgPool2d(kernel_size=2, stride=2)
            self.convh = HexConv2d(in_channels, hf_ch_out, self.hf_level, stride=1)
            self.avg_pool = UnfoldAvgPool2d(kernel_size=2, stride=2)
            self.convl = HexConv2d(in_channels, lf_ch_out, self.lf_level, stride=1)

        elif type == 'last':
            self.hf_level = level if stride == 1 else level - 1
            if stride == 2:
                self.downsample = UnfoldAvgPool2d(kernel_size=2, stride=2)
            self.convh = HexConv2d(hf_ch_in, out_channels, self.hf_level, stride=1)
            self.convl = HexConv2d(lf_ch_in, out_channels, self.hf_level, stride=1)
            self.upsample = UnfoldUpsample()
        else:
            self.hf_level = level if stride == 1 else level - 1
            self.lf_level = self.hf_level - 1
            if stride == 2:
                self.downsample = UnfoldAvgPool2d(kernel_size=2, stride=2)
                self.l2h = HexConv2d(lf_ch_in, hf_ch_out, self.hf_level, stride=1)
            else:
                self.l2h = HexConv2d(lf_ch_in, hf_ch_out, self.lf_level, stride=1)

            self.l2l = HexConv2d(lf_ch_in, lf_ch_out, self.lf_level, stride=1)

            self.h2l = HexConv2d(hf_ch_in, lf_ch_out, self.lf_level, stride=1)
            self.h2h = HexConv2d(hf_ch_in, hf_ch_out, self.hf_level, stride=1)
            self.upsample = UnfoldUpsample()
            self.avg_pool = UnfoldAvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        if self.type == 'first':
            if self.stride == 2:
                x = self.downsample(x)

            hf = self.convh(x)
            lf = self.avg_pool(x)
            lf = self.convl(lf)

            return hf, lf
        elif self.type == 'last':
            hf, lf = x
            if self.stride == 2:
                hf = self.downsample(hf)
                return self.convh(hf) + self.convl(lf)
            else:
                return self.convh(hf) + self.convl(self.upsample(lf))
        else:
            hf, lf = x
            if self.stride == 2:
                hf = self.downsample(hf)
                return self.h2h(hf) + self.l2h(lf), \
                       self.l2l(self.avg_pool(lf)) + self.h2l(self.avg_pool(hf))
            else:
                return self.h2h(hf) + self.upsample(self.l2h(lf)), self.l2l(lf) + self.h2l(self.avg_pool(hf))


class OctHexConvBNrelu(nn.Module):
    def __init__(self, in_channels, out_channels, level, stride=1, relu=True,
                 alpha_in=None, alpha_out=None, type='normal'):
        super(OctHexConvBNrelu, self).__init__()

        if alpha_in is None and ALPHA is not None:
            alpha_in = ALPHA
        if alpha_out is None and ALPHA is not None:
            alpha_out = ALPHA

        # self.stride = stride
        # self.type = type
        hf_ch_in = int(in_channels * (1 - alpha_in))
        hf_ch_out = int(out_channels * (1 - alpha_out))
        lf_ch_in = in_channels - hf_ch_in
        lf_ch_out = out_channels - hf_ch_out

        self.conv = OctHexConv2d(in_channels, out_channels, level, stride, alpha_in, alpha_out, type)
        self.hf_bn = UnfoldBatchNorm2d(hf_ch_out)
        self.lf_bn = UnfoldBatchNorm2d(lf_ch_out)

        self.relu = UnfoldReLU() if relu else None

    def forward(self, x):
        hf, lf = self.conv(x)
        hf = self.hf_bn(hf)
        lf = self.lf_bn(lf)

        if self.relu is not None:
            hf = self.relu(hf)
            lf = self.relu(lf)

        return hf, lf


class OctResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, level, bias=False):
        super(OctResBlock, self).__init__()

        hf_ch_in = int(in_channels * (1 - ALPHA))
        hf_ch_out = int(out_channels * (1 - ALPHA))
        lf_ch_in = in_channels - hf_ch_in
        lf_ch_out = out_channels - hf_ch_out

        self.conv1_hf = nn.Sequential(
            UnfoldConv2d(hf_ch_in, hf_ch_out, 1, bias=bias), UnfoldMaxPool2d(2),
            UnfoldBatchNorm2d(hf_ch_out), UnfoldReLU())
        self.conv1_lf = nn.Sequential(
            UnfoldConv2d(lf_ch_in, lf_ch_out, 1, bias=bias), UnfoldMaxPool2d(2),
            UnfoldBatchNorm2d(lf_ch_out), UnfoldReLU())

        self.conv2 = OctHexConvBNrelu(out_channels, out_channels, level - 1, stride=1)
        self.conv3_hf = nn.Sequential(
            UnfoldConv2d(hf_ch_out, hf_ch_out, 1, bias=bias), UnfoldBatchNorm2d(hf_ch_out))
        self.conv3_lf = nn.Sequential(
            UnfoldConv2d(lf_ch_out, lf_ch_out, 1, bias=bias), UnfoldBatchNorm2d(lf_ch_out))

        self.downsample_hf = nn.Sequential(
            UnfoldConv2d(hf_ch_in, hf_ch_out, 1, bias=bias), UnfoldMaxPool2d(2),
            UnfoldBatchNorm2d(hf_ch_out))
        self.downsample_lf = nn.Sequential(
            UnfoldConv2d(lf_ch_in, lf_ch_out, 1, bias=bias), UnfoldMaxPool2d(2),
            UnfoldBatchNorm2d(lf_ch_out))

        self.relu = UnfoldReLU()

    def forward(self, x):
        hf, lf = x
        identity_hf = hf
        identity_lf = lf

        hf = self.conv1_hf(hf)
        lf = self.conv1_lf(lf)

        hf, lf = self.conv2((hf, lf))
        hf = self.conv3_hf(hf)
        lf = self.conv3_lf(lf)

        identity_hf = self.downsample_hf(identity_hf)
        identity_lf = self.downsample_lf(identity_lf)
        for i in range(5):
            hf[i] += identity_hf[i]
            lf[i] += identity_lf[i]

        hf = self.relu(hf)
        lf = self.relu(lf)

        return hf, lf


class OctHexRUNet_C(nn.Module):
    """ OctHexRUNet-C """

    def __init__(self, in_channels):
        super(OctHexRUNet_C, self).__init__()

        mid_channel = 192

        self.conv1 = OctHexConvBNrelu(in_channels, 16, level=4, type='first')
        self.block1 = OctResBlock(16, 64, level=4)
        self.block2 = OctResBlock(64, mid_channel, level=3)
        self.fusion = OctHexConv2d(mid_channel, mid_channel, level=2, type='last')
        self.fc = nn.Linear(mid_channel, 10)

    def forward(self, batch):
        out = self.conv1(batch)
        out = self.block1(out)
        out = self.block2(out)
        out = self.fusion(out)

        # Max pooling. I'm not sure if it's correct
        cat_feat = [out[i] for i in range(5)]
        cat_feat = torch.cat(cat_feat, axis=3)
        b, c, h, w = out[0].shape
        cat_feat = cat_feat.view(b, c, -1)
        flatten_feat = torch.max(cat_feat, dim=2)[0]

        out = self.fc(flatten_feat)

        return out


class OctHexRUNet_C_2(nn.Module):
    """ OctHexRUNet-C """

    def __init__(self, in_channels):
        super(OctHexRUNet_C_2, self).__init__()

        mid_channel = 256

        self.conv1 = OctHexConvBNrelu(in_channels, 16, level=4, type='first')
        self.block1 = OctResBlock(16, 64, level=4)
        self.block2 = OctResBlock(64, mid_channel, level=3)
        self.pool = UnfoldAvgPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(mid_channel, 10)

    def forward(self, batch):
        out = self.conv1(batch)
        out = self.block1(out)
        out = self.block2(out)
        hf, lf = out
        hf = self.fusion(hf)
        out = [torch.cat([hf[i], lf[i]], dim=1) for i in range(5)]

        # Max pooling. I'm not sure if it's correct
        cat_feat = [out[i] for i in range(5)]
        cat_feat = torch.cat(cat_feat, axis=3)
        b, c, h, w = out[0].shape
        cat_feat = cat_feat.view(b, c, -1)
        flatten_feat = torch.max(cat_feat, dim=2)[0]

        out = self.fc(flatten_feat)

        return out
