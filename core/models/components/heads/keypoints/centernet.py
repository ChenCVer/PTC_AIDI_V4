# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from core.models.registry import HEADS
from core.models.components.heads.base import BaseHead
from mmcv.cnn import constant_init, kaiming_init
from torch.nn.modules.batchnorm import _BatchNorm
from mmcv.cnn import ConvModule, build_activation_layer, \
                     build_norm_layer, build_upsample_layer

__all__ = ['CenternetHead_hm', ]


@HEADS.register_module()
class CenternetHead_hm(BaseHead):
    def __init__(self,
                 in_channels,
                 num_classes=1,
                 num_layers=3,
                 num_filters=[256, 128, 64],
                 num_kernels=[4, 4, 4],
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True, momentum=0.1),
                 act_cfg=dict(type='ReLU', inplace=True),
                 losser=None,
                 metricer=None):

        super(CenternetHead_hm, self).__init__(losser, metricer)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self._init_layers(num_layers, num_filters, num_kernels)

    def _init_layers(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        self.up_samples = nn.Sequential()
        for i in range(num_layers):
            # conv
            if i == 0:
                conv = ConvModule(self.in_channels, num_filters[i],
                                  kernel_size=3, stride=1, padding=1,
                                  norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
            else:
                conv = ConvModule(num_filters[i-1], num_filters[i],
                                  kernel_size=3, stride=1, padding=1,
                                  norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
            self.up_samples.add_module("conv_{}".format(i), conv)

            # deconv
            kernel, padding, output_padding = self._get_deconv_cfg(num_kernels[i])
            planes = num_filters[i]
            deconv_cfg = dict(type="deconv", in_channels=planes,
                              out_channels=planes, kernel_size=kernel, stride=2,
                              padding=padding, output_padding=output_padding, bias=False)
            deconv = build_upsample_layer(deconv_cfg)
            self.up_samples.add_module("deconv_{}".format(i), deconv)

            # bn
            _, norm = build_norm_layer(self.norm_cfg, planes)
            self.up_samples.add_module("norm_{}".format(i), norm)

            # act
            act = build_activation_layer(self.act_cfg)
            self.up_samples.add_module("act_{}".format(i), act)

        # hm_head
        self.hm_preds = nn.Sequential()
        conv_bridge = nn.Conv2d(
            num_filters[-1],
            num_filters[-1],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True)
        self.hm_preds.add_module("conv_bridge", conv_bridge)
        act = nn.ReLU(inplace=True)
        self.hm_preds.add_module("act", act)
        conv_pred = nn.Conv2d(
            num_filters[-1],
            self.num_classes,
            kernel_size=1, stride=1,
            padding=0, bias=True)
        self.hm_preds.add_module("conv_pred", conv_pred)

    def init_weights(self):
        for m in self.modules():
            # init deconv weights
            if isinstance(m, nn.ConvTranspose2d):
                kaiming_init(m, mode='fan_out')
            # init conv weights
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            # init BN weights
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)

    def _get_deconv_cfg(self, deconv_kernel):
        padding, output_padding = 0, 0

        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def forward(self, inputs):
        if isinstance(inputs, tuple):
            if len(inputs) == 1:
                out = self.up_samples(inputs[0])
                out = self.hm_preds(out)
            else:
                # todo: 暂时不考虑多尺度
                pass
        elif isinstance(inputs, torch.Tensor):
            out = self.up_samples(inputs)
            out = self.hm_preds(out)
        else:
            raise TypeError('neck inputs should be tuple or torch.tensor')

        return out
