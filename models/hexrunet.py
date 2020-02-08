import torch
import torch.nn as nn

from models.unfold_nn import HexConv2d, UnfoldReLU, UnfoldBatchNorm2d, UnfoldMaxPool2d, UnfoldConv2d


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, level, bias=False):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Sequential(
            UnfoldConv2d(in_channels, out_channels, 1, bias=bias), UnfoldMaxPool2d(2),
            UnfoldBatchNorm2d(out_channels), UnfoldReLU(inplace=True))
        self.conv2 = nn.Sequential(
            HexConv2d(out_channels, out_channels, level - 1, 1, bias=bias),
            UnfoldBatchNorm2d(out_channels), UnfoldReLU(inplace=True))
        self.conv3 = nn.Sequential(
            UnfoldConv2d(out_channels, out_channels, 1, bias=bias), UnfoldBatchNorm2d(out_channels))

        self.downsample = nn.Sequential(
            UnfoldConv2d(in_channels, out_channels, 1, bias=bias), UnfoldMaxPool2d(2),
            UnfoldBatchNorm2d(out_channels))

        self.relu = UnfoldReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        identity = self.downsample(identity)
        for i in range(5):
            out[i] += identity[i]

        out = self.relu(out)
        return out


class HexRUNet_C(nn.Module):
    """ HexRUNet-C proposed in [1].

    References
    ----------
    [1] orientation-aware semantic segmentation on icosahedron spheres, ICCV2019

    """

    def __init__(self, in_channels):
        super(HexRUNet_C, self).__init__()
        self.conv1 = nn.Sequential(
            HexConv2d(in_channels, 16, level=4, stride=1), UnfoldReLU(inplace=True), UnfoldBatchNorm2d(16))
        self.block1 = ResBlock(16, 64, level=4)
        self.block2 = ResBlock(64, 256, level=3)
        self.fc = nn.Linear(256, 10)

    def forward(self, batch):
        out = self.conv1(batch)
        out = self.block1(out)
        out = self.block2(out)

        # Max pooling. I'm not sure if it's correct
        cat_feat = [out[i] for i in range(5)]
        cat_feat = torch.cat(cat_feat, axis=3)
        b, c, h, w = out[0].shape
        cat_feat = cat_feat.view(b, c, -1)
        flatten_feat = torch.max(cat_feat, dim=2)[0]

        out = self.fc(flatten_feat)

        return out
