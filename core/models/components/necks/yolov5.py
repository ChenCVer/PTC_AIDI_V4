import math
import torch.nn as nn
from ..brick.focus import Concat
from core.models.registry import NECKS
from ..backbones.det.yolov5darknet import BottleneckCSP
from torch.nn.modules.batchnorm import _BatchNorm
from mmcv.cnn import ConvModule, constant_init, kaiming_init


@NECKS.register_module()
class Yolov5Neck(nn.Module):

    def __init__(self,
                 depth_multiple,
                 width_multiple,
                 in_channels=1024,
                 out_channels=[512, 256, 256, 512],
                 shortcut=[False, False, False, False],
                 bottle_depths=[3, 3, 3, 3],
                 upsampling_mode='nearest',
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1),):

        super(Yolov5Neck, self).__init__()
        assert upsampling_mode.lower() in ['nearest', 'linear', 'bilinear', 'bicubic', 'trilinear'], \
            "upsampling mode just support ['nearest','linear','bilinear', 'bicubic','trilinear']"
        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.depth_multiple = depth_multiple
        self.width_multiple = width_multiple
        self.shortcut = shortcut
        self.bottle_depths = bottle_depths
        self.out_channels = out_channels
        expand_in_channels = math.ceil(in_channels * self.width_multiple / 8) * 8
        self.concats = []
        self.module_list = nn.ModuleList()
        # bottle_depths同样表示的是CSP中Res_Unit的个数.
        for idx, depth in enumerate(bottle_depths):
            hidden_channels = self.out_channels[idx]
            num_resblock = max(round(depth * self.depth_multiple), 1) if depth > 1 else depth
            if idx < 2:
                out_channels = math.ceil(hidden_channels * self.width_multiple / 8) * 8
                # build CBL module.
                self.module_list.add_module(f'conv_1x1_{idx}',
                                            ConvModule(expand_in_channels, out_channels,
                                                       kernel_size=1, stride=1, **cfg))
                self.concats.append(len(self.module_list) - 1)
                # upsampling.
                self.module_list.add_module(f'upsample_{idx}',
                                            nn.Upsample(scale_factor=2,
                                                        mode=upsampling_mode.lower()))
                # concat.
                self.module_list.add_module(f'concat_{idx}', Concat())
                expand_in_channels = out_channels * 2
                # bulid CSP module.
                self.module_list.add_module(f'csp_module_{idx}',
                                            BottleneckCSP(expand_in_channels, out_channels,
                                                          num_resblock=num_resblock,
                                                          shortcut=self.shortcut[idx]))
                expand_in_channels = out_channels
            else:
                out_channels = math.ceil(hidden_channels * self.width_multiple / 8) * 8
                # build CBL module.
                self.module_list.add_module(f'conv_3x3_{idx}',
                                            ConvModule(expand_in_channels, out_channels,
                                                       kernel_size=3, stride=2, padding=1, **cfg))
                # concat.
                self.module_list.add_module(f'concat_{idx}', Concat())
                expand_in_channels = out_channels * 2
                out_channels = math.ceil(hidden_channels * 2 * self.width_multiple / 8) * 8
                # bulid CSP module.
                self.module_list.add_module(f'csp_module_{idx}',
                                            BottleneckCSP(expand_in_channels, out_channels,
                                                          num_resblock=num_resblock,
                                                          shortcut=self.shortcut[idx]))
                expand_in_channels = out_channels

    def forward(self, x):

        y, x, c_idx, outs = x[:2], x[-1], 0, []

        for idx, module in enumerate(self.module_list):
            if isinstance(module, Concat):
                x = [x, y[c_idx - 1 if c_idx % 2 != 0 else c_idx + 1]]
                c_idx += 1
            x = module(x)
            if idx in self.concats:
                y.append(x)
            if isinstance(module, BottleneckCSP):
                outs.append(x)

        if len(outs) == 0:
            outs.append(x)

        return list(reversed(outs[1:]))

    def init_weights(self):
        """
        init Conv2D in BottleneckCSP.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)
