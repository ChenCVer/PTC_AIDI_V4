import math
import torch
import logging
from torch import nn
from ...brick import SPP
from ...brick import Focus
from mmcv.runner import load_checkpoint
from core.models.registry import BACKBONES
from torch.nn.modules.batchnorm import _BatchNorm
from mmcv.cnn import ConvModule, constant_init, kaiming_init


@BACKBONES.register_module()
class YOLOv5Darknet(nn.Module):
    def __init__(self,
                 depth_multiple,  # 控制网路深度, 残差单元的个数, 作用于backbone和neck的CSP1_X, CSP2_X
                 width_multiple,  # 控制网络通道个数, 只作用于backbone的CBL
                 focus,
                 in_channels=3,
                 frozen_stages=-1,
                 bottle_depths=[3, 9, 9, 3],
                 out_channels=[128, 256, 512, 1024],
                 spp=[5, 9, 13],
                 shortcut=[True, True, True, False],
                 out_indices=(2, 3, 4,),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
                 norm_eval=True):

        super(YOLOv5Darknet, self).__init__()
        self.depth_multiple = depth_multiple
        self.width_multiple = width_multiple
        self.bottle_depths = bottle_depths
        self.out_channels = out_channels
        self.frozen_stages = frozen_stages
        assert len(self.bottle_depths) == len(self.out_channels), \
            'len(self.bottle_depths) must = len(self.bottle_out_channels)'

        _in_channels = in_channels
        # build focus环节.
        if focus is not None:
            out_channel_1 = focus[0]
            # 通过width_multiple参数控制网络宽度.
            out_channel_1 = math.ceil(out_channel_1 * self.width_multiple / 8) * 8
            self.focus = Focus(in_channels=in_channels,
                               out_channels=out_channel_1,
                               kernel_size=focus[1],
                               stride=focus[2],
                               conv_cfg=None,
                               norm_cfg=norm_cfg,
                               act_cfg=act_cfg)

            _in_channels = out_channel_1

        self.out_indices = []
        self.module_list = nn.ModuleList()
        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        # 交替构建conv3x3和CSP环节.
        # 需要注意的是: len(self.bottle_depths)=4, 包含backbone中的3个CSP1_X和neck中的第一个CSP2_X.
        # 其中bottle_depths表示每个CSP组件中的残差组件的个数.比如说: CSP1_3表示CSP中有3个残差组件.
        for i, depth in enumerate(self.bottle_depths):
            out_channel_2 = self.out_channels[i]
            out_channel_2 = math.ceil(out_channel_2 * self.width_multiple / 8) * 8
            num_resblocks = max(round(depth * self.depth_multiple), 1) if depth > 1 else depth
            # bulid CBL module.
            self.module_list.add_module(f'conv_3x3_{i}',
                                        ConvModule(_in_channels, out_channel_2,
                                                   kernel_size=3, stride=2, padding=1, **cfg))
            _in_channels = out_channel_2

            # bulid spp module.
            if i == len(self.bottle_depths) - 1 and spp is not None:
                self.module_list.add_module(f'spp_{i}', SPP(_in_channels, out_channel_2, k=spp))
                _in_channels = out_channel_2

            # build CSP module, n is behalf of the repeated times of Res unit.
            self.module_list.add_module(f'csp_module_{i}',
                                        BottleneckCSP(_in_channels, out_channel_2,
                                                      num_resblock=num_resblocks,
                                                      shortcut=shortcut[i]))
            _in_channels = out_channel_2

            if i + 1 in out_indices:
                self.out_indices.append(len(self.module_list) - 1)

        self.norm_eval = norm_eval

    def forward(self, x):
        out = self.focus(x)
        outs = []
        for idx, _m in enumerate(self.module_list):
            out = _m(out)
            if idx in self.out_indices:
                outs.append(out)

        # if out_indices is empty, then the lastest output of
        # backbone will come to be the default.
        if len(outs) == 0:
            outs.append(out)

        return outs

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for i in range(self.frozen_stages):
                m = getattr(self, self.cr_blocks[i])
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        super(YOLOv5Darknet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()


class BottleneckCSP(nn.Module):
    """
    BottleneckCSP is illustrated as follow below:
    CSP1_X = --+->CBL(main) -> ResUnit_X -> CBL(post) -+-> concat -> CBL(final) ->
               |                                       |
               +---------------> CBL(short) -----------+
    This class makes the conv_res_block in YOLO v4. It has CSP integrated,
    hence different from the regular conv_res_block build with
    `make_conv_res_block`.
    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        num_resblock (int): The number of ResBlocks.
        groups (int): groups of conv.
        expansion(float): base of net depth.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN', requires_grad=True)
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='LeakyReLU', negative_slope=0.1).
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_resblock=1,
                 shortcut=True,
                 groups=1,
                 expansion=0.5,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1)):

        super(BottleneckCSP, self).__init__()
        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        hidden_channels = int(out_channels * expansion)
        self.mainconv = ConvModule(
            in_channels, hidden_channels, kernel_size=1, stride=1, **cfg)
        self.shortconv = nn.Conv2d(
            in_channels, hidden_channels, 1, stride=1, bias=False)
        self.postconv = nn.Conv2d(
            hidden_channels, hidden_channels, 1, stride=1, bias=False)
        self.finalconv = ConvModule(
            2 * hidden_channels, out_channels, kernel_size=1, stride=1, **cfg)
        self.norm = nn.BatchNorm2d(2 * hidden_channels)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.moudle_list = nn.Sequential(*[Bottleneck(hidden_channels, hidden_channels,
                                                      shortcut, groups, expansion=1.0)
                                                      for _ in range(num_resblock)])

    def forward(self, input):
        # main
        x_main = self.mainconv(input)
        x_main = self.moudle_list(x_main)
        x_main = self.postconv(x_main)
        # short
        x_short = self.shortconv(input)
        # concat
        x_final = torch.cat((x_main, x_short), dim=1)
        x_final = self.norm(x_final)
        x_final = self.act(x_final)
        x_final = self.finalconv(x_final)

        return x_final


class Bottleneck(nn.Module):
    """
    This class makes the conv_res_block in YOLO v4. It has CSP integrated,
    hence different from the regular conv_res_block build with
    `make_conv_res_block`.
    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        shortcut (bool): wether requires a shortcut.
        groups (int): groups of conv.
        expansion(float): base of net depth.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN', requires_grad=True)
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='LeakyReLU', negative_slope=0.1).
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 shortcut=True,
                 groups=1,
                 expansion=0.5,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1)):

        super(Bottleneck, self).__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv1 = ConvModule(
            in_channels, hidden_channels, kernel_size=1, stride=1, **cfg)
        self.conv2 = ConvModule(
            hidden_channels, out_channels, kernel_size=3, stride=1, padding=1, **cfg)
        self.make_shortcut = shortcut and in_channels == out_channels

    def forward(self, x):
        if self.make_shortcut:
            out = self.conv1(x)
            out = self.conv2(out)
            out = out + x
            return out
        else:
            out = self.conv1(x)
            out = self.conv2(out)
            return out