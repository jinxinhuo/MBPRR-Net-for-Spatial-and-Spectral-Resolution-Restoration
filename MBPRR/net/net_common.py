import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule


def gcd(n, m):
    if m == 0:
        return n
    return gcd(m, n % m)


def default_conv(ch_in, ch_out, k_size=(3, 3), stride=1, bias=True, group=False):
    if isinstance(k_size, tuple):
        kernel_size1, kernel_size2 = k_size
        if group:
            return nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=k_size, stride=stride,
                             padding=(kernel_size1 // 2, kernel_size2 // 2), bias=bias, groups=gcd(ch_in, ch_out))
        else:
            return nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=k_size, stride=stride,
                             padding=(kernel_size1 // 2, kernel_size2 // 2), bias=bias)
    if group:
        return nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=k_size, stride=stride,
                         padding=(k_size // 2), bias=bias, groups=gcd(ch_in, ch_out))
    else:
        return nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=k_size, stride=stride,
                         padding=(k_size // 2), bias=bias)


def Mish(input):
    return input * torch.tanh(F.softplus(input))


class RFB_Block2(nn.Module):
    def __init__(self, ch_in, ch_out, groups=False):
        super(RFB_Block2, self).__init__()

        self.conv1_1 = default_conv(ch_in=ch_in, ch_out=ch_in, k_size=1, stride=1, bias=True, group=groups)
        self.conv1_2 = default_conv(ch_in=ch_in, ch_out=ch_in, k_size=1, stride=1, bias=True, group=groups)

        self.conv2_1 = default_conv(ch_in=ch_in, ch_out=ch_in, k_size=1, stride=1, bias=True, group=groups)
        self.conv2_2 = default_conv(ch_in=ch_in, ch_out=ch_in, k_size=3, stride=1, bias=True, group=groups)

        self.conv3_1 = default_conv(ch_in=ch_in, ch_out=ch_in, k_size=1, stride=1, bias=True, group=groups)
        self.conv3_2 = default_conv(ch_in=ch_in, ch_out=ch_in, k_size=3, stride=1, bias=True, group=groups)

        self.conv4_1 = default_conv(ch_in=ch_in, ch_out=ch_in, k_size=1, stride=1, bias=True, group=groups)
        self.conv4_2 = default_conv(ch_in=ch_in, ch_out=ch_in, k_size=3, stride=1, bias=True, group=groups)

        self.conv1_sum = default_conv(ch_in=ch_in * 4, ch_out=ch_out, k_size=1, stride=1, bias=True)

    def forward(self, x):
        b1 = Mish(self.conv1_1(x))
        b1 = Mish(self.conv1_2(b1)) + b1

        b2 = Mish(self.conv2_1(x))
        b2 = Mish(self.conv2_2(b2)) + b2
        b2 = Mish(self.conv2_2(b2)) + b2

        b3 = Mish(self.conv3_1(x))
        b3 = Mish(self.conv3_2(b3)) + b3
        b3 = Mish(self.conv3_2(b3)) + b3
        b3 = Mish(self.conv3_2(b3)) + b3

        b4 = Mish(self.conv4_1(x))
        b4 = Mish(self.conv4_2(b4)) + b4

        sum_4 = torch.cat([b1, b2, b3, b4], dim=1)
        sum_4 = Mish(self.conv1_sum(sum_4))

        return sum_4


class ConvUpsampler(nn.Sequential):
    def __init__(self, ch_in, ch_out, groups=False):
        super(ConvUpsampler, self).__init__()
        self.conv1 = default_conv(ch_in=ch_in, ch_out=ch_out * 4, k_size=3, group=groups)
        self.ps2 = nn.PixelShuffle(2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.ps2(x)
        return x


class involution(nn.Module):

    def __init__(self,
                 channels,
                 kernel_size,
                 stride):
        super(involution, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.channels = channels
        reduction_ratio = 4
        self.group_channels = 16
        self.groups = self.channels // self.group_channels
        self.conv1 = ConvModule(
            in_channels=channels,
            out_channels=channels // reduction_ratio,
            kernel_size=1,
            conv_cfg=None,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'))
        self.conv2 = ConvModule(
            in_channels=channels // reduction_ratio,
            out_channels=kernel_size ** 2 * self.groups,
            kernel_size=1,
            stride=1,
            conv_cfg=None,
            norm_cfg=None,
            act_cfg=None)
        if stride > 1:
            self.avgpool = nn.AvgPool2d(stride, stride)
        self.unfold = nn.Unfold(kernel_size, 1, (kernel_size - 1) // 2, stride)

    def forward(self, x):
        weight = self.conv2(self.conv1(x if self.stride == 1 else self.avgpool(x)))
        b, c, h, w = weight.shape
        weight = weight.view(b, self.groups, self.kernel_size ** 2, h, w).unsqueeze(2)
        out = self.unfold(x).view(b, self.groups, self.group_channels, self.kernel_size ** 2, h, w)
        out = (weight * out).sum(dim=3).view(b, self.channels, h, w)
        return out


class involution2(nn.Module):

    def __init__(self,
                 channels,
                 kernel_size,
                 stride, reduction=1):
        super(involution2, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.channels = channels
        reduction_ratio = 2
        self.group_channels = 4
        self.groups = self.channels // self.group_channels
        self.conv1 = ConvModule(
            in_channels=channels,
            out_channels=channels // reduction_ratio,
            kernel_size=1,
            conv_cfg=None,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'))
        self.conv2 = ConvModule(
            in_channels=channels // reduction_ratio,
            out_channels=kernel_size ** 2 * self.groups,
            kernel_size=1,
            stride=1,
            conv_cfg=None,
            norm_cfg=None,
            act_cfg=None)
        if stride > 1:
            self.avgpool = nn.AvgPool2d(stride, stride)
        self.unfold = nn.Unfold(kernel_size, 1, (kernel_size - 1) // 2, stride)

    def forward(self, x, y):
        weight = self.conv2(self.conv1(x if self.stride == 1 else self.avgpool(x)))
        b, c, h, w = weight.shape
        weight = weight.view(b, self.groups, self.kernel_size ** 2, h, w).unsqueeze(2)
        out = self.unfold(y).view(b, self.groups, self.group_channels, self.kernel_size ** 2, h, w)
        out = (weight * out).sum(dim=3).view(b, self.channels, h, w)
        return out

