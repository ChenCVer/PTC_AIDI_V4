# -*- coding:utf-8 -*-
import torch.nn as nn
from core.models.registry import BACKBONES
from ...brick.unet_parts import InConv, DownSample

__all__ = ['UnetEncoder']


@BACKBONES.register_module()
class UnetEncoder(nn.Module):
    def __init__(self,
                 input_channel=3,
                 root_channels=16,
                 layer_num=4,
                 kernel_size=3,
                 use_double_conv=True,
                 shortcut=False,
                 dowmsample_style='maxpool',
                 norm=nn.BatchNorm2d):

        super(UnetEncoder, self).__init__()

        self.shortcut = shortcut  # shortcut
        self.inc = InConv(input_channel, root_channels,
                          kernel_size, use_double_conv, norm)
        downsample_list = []
        for i in range(1, layer_num + 1):
            downsample_list.append(DownSample(root_channels * 2 ** (i - 1),
                                              root_channels * 2 ** i,
                                              kernel_size, use_double_conv,
                                              norm, dowmsample_style))
        self.down_convs = nn.ModuleList(downsample_list)

    def forward(self, input_tensor):
        output = []
        x = self.inc(input_tensor)
        output.append(x)
        x = self.down_convs[0](x)
        output.append(x)
        for i in range(1, len(self.down_convs)):
            x = self.down_convs[i](x)
            output.append(x)
        return output

    def init_weights(self, pretrained=None):
        """
        Note:
            segmentation training generally does not require (rarely required) pre-training
            weights, especially in the industrial field.
        """
        pass
