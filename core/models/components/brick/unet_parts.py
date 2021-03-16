# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    (conv => BN => ReLU) * 2, stride of conv is 1.
    """

    def __init__(self, in_ch, out_ch, kernel_size, norm=nn.BatchNorm2d):
        super(DoubleConv, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding),
            norm(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size, padding=padding),
            norm(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class ResDoubleConv(nn.Module):
    #
    #   ----->conv+bn+relu--->conv+bn----+-->relu--->
    #     |                              |
    #     ┕------------>conv+bn----------┙
    def __init__(self, in_ch, out_ch, kernel_size, norm=nn.BatchNorm2d):
        super(ResDoubleConv, self).__init__()
        padding = (kernel_size - 1) // 2

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding),
            norm(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size, padding=padding),
            norm(out_ch),
        )
        # 由于输入与输出单元通道数不一样, 这里需要conv_1x1卷积改变维度.
        self.identity_block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            norm(out_ch),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        residual = self.identity_block(x)
        out = self.conv_block(x)
        out = out + residual

        out = self.relu(out)

        return out


class SingelConv(nn.Module):
    """(conv => BN => ReLU) """

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, norm=nn.BatchNorm2d):
        super(SingelConv, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding),
            norm(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class ResSingleConv(nn.Module):
    #
    #   ------->conv+bn-----+--->relu--->
    #     |                 |
    #     ┕----->conv+bn----┙
    def __init__(self, in_ch, out_ch, kernel_size, norm=nn.BatchNorm2d):
        super(ResSingleConv, self).__init__()
        padding = (kernel_size - 1) // 2

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding),
            norm(out_ch),
        )
        # conv_1x1
        self.identity_block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            norm(out_ch),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        residual = self.identity_block(x)
        out = self.conv_block(x)
        out = out + residual

        out = self.relu(out)

        return out


class InConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, use_double_conv, norm=nn.BatchNorm2d):
        super(InConv, self).__init__()
        if use_double_conv:
            self.conv = DoubleConv(in_ch, out_ch, kernel_size, norm)
        else:
            self.conv = SingelConv(in_ch, out_ch, kernel_size)

    def forward(self, x):
        x = self.conv(x)
        return x


class DownSample(nn.Module):

    downsample_style = ['maxpool', 'conv_down']

    def __init__(self, in_ch, out_ch, kernel_size,
                 use_double_conv, norm=nn.BatchNorm2d,
                 dowmsample_style='maxpool',
                 shortcut=False):

        super(DownSample, self).__init__()

        if dowmsample_style not in self.downsample_style:
            raise KeyError(f'invalid down_sample: {dowmsample_style} for DownSample')

        # downsample style
        if dowmsample_style == "maxpool":
            down_module = nn.MaxPool2d(2)
        else:
            # conv stride/2 -> bn -> relu
            down_module = SingelConv(in_ch, in_ch, kernel_size,
                                     stride=2, norm=norm)

        # shortcut
        if shortcut:
            # double conv
            if use_double_conv:
                ConvModule = ResDoubleConv(in_ch, out_ch, kernel_size, norm)
            else:
                ConvModule = ResSingleConv(in_ch, out_ch, kernel_size, norm)
        else:
            if use_double_conv:
                ConvModule = DoubleConv(in_ch, out_ch, kernel_size, norm)
            else:
                ConvModule = SingelConv(in_ch, out_ch, kernel_size, norm=norm)

        self.mpconv = nn.Sequential(
                down_module,
                ConvModule,
            )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size,
                 use_double_conv, bilinear=True,
                 align_corners=True,
                 norm=nn.BatchNorm2d,
                 shortcut=False):
        super(UpSample, self).__init__()
        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        # pytorch的一切上采样操作, 在被转变为ONNX时, 都会被翻译为Resize, 但是ONNX里面的resize
        # 要求output shape必须为常量(即tuple of int), 但是下面的写法nn.Upsample,可能会导致
        # 输出output shape不是常量. 具体解答参见:
        # https://zhuanlan.zhihu.com/p/286298001
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=align_corners)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2 + out_ch // 2, in_ch // 2 + out_ch // 2, 2, stride=2)

        if shortcut:
            if use_double_conv:
                self.conv = ResDoubleConv(in_ch, out_ch, kernel_size, norm)
            else:
                self.conv = ResSingleConv(in_ch, out_ch, kernel_size, norm=norm)
        else:
            if use_double_conv:
                self.conv = DoubleConv(in_ch, out_ch, kernel_size, norm)
            else:
                self.conv = SingelConv(in_ch, out_ch, kernel_size, norm=norm)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class OutConvRefine(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, use_double_conv):
        super(OutConvRefine, self).__init__()
        if use_double_conv:
            self.conv1 = DoubleConv(in_ch, 16, kernel_size)
        else:
            self.conv1 = SingelConv(in_ch, 16, kernel_size)
        self.conv2 = nn.Conv2d(16, out_ch, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
