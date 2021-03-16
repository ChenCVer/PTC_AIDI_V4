import torch
import logging
import torch.nn as nn
from mmcv.cnn import ConvModule, constant_init, kaiming_init
from mmcv.runner import load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm
from core.models.components.brick import DropBlock2D_Pool
from core.models.registry import BACKBONES

__all__ = ["Darknet",
           ]


class ResBlock(nn.Module):
    """The basic residual block used in Darknet. Each ResBlock consists of two
    ConvModules and the input is added to the final output. Each ConvModule is
    composed of Conv, BN, and LeakyReLU. In YoloV3 paper, the first convLayer
    has half of the number of the filters as much as the second convLayer. The
    first convLayer has filter size of 1x1 and the second one has the filter
    size of 3x3.

    Args:
        in_channels (int): The input channels. Must be even.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN', requires_grad=True)
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='LeakyReLU', negative_slope=0.1).
    """

    def __init__(self,
                 in_channels,
                 yolo_version='v3',
                 is_dropblock=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1)):

        super(ResBlock, self).__init__()
        if yolo_version not in ('v3', 'v4'):
            raise NotImplementedError('Only YOLO v3 and v4 are supported.')

        self.is_dropblock = is_dropblock
        if is_dropblock:
            self.dropblock = DropBlock2D_Pool()

        # shortcut
        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        if yolo_version == 'v3':
            assert in_channels % 2 == 0  # ensure the in_channels is even
            mid_channels = in_channels // 2
        else:
            mid_channels = in_channels

        # conv_1x1
        self.conv1 = ConvModule(in_channels, mid_channels, 1, **cfg)
        # conv_3x3
        self.conv2 = ConvModule(mid_channels, in_channels, 3, padding=1, **cfg)

    def forward(self, x):
        residual = x
        # dropblock
        if self.is_dropblock:
            x = self.dropblock(x)
        # conv_1x1
        out = self.conv1(x)
        # conv_3x3
        out = self.conv2(out)
        # dropblock
        if self.is_dropblock:
            out = self.dropblock(out)
        # res
        out = out + residual

        return out


@BACKBONES.register_module()
class Darknet(nn.Module):
    """Darknet backbone for yolov3 and yolov4.
    Args:
        depth (int): Depth of Darknet. Currently only support 53.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Default: -1.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        with_csp (bool): Whether the Darknet uses csp (cross stage partial
            network). This is a feature of YOLO v4, see details at
            `https://arxiv.org/abs/1911.11929`_ Default: False.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN', requires_grad=True)
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='LeakyReLU', negative_slope=0.1).
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.

    Example:
        >>> import torch
        >>> self = Darknet(depth=53)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 416, 416)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        ...
        (1, 256, 52, 52)
        (1, 512, 26, 26)
        (1, 1024, 13, 13)
    """

    # Dict(depth: (layers, channels))
    arch_settings = {
        53: ((1, 2, 8, 8, 4), ((32, 64), (64, 128), (128, 256), (256, 512), (512, 1024)))
    }
    # dropblock stage
    dropblock_stage = [0, 1, 2, 3, 4]

    def __init__(self,
                 depth=53,
                 out_indices=(3, 4, 5),
                 frozen_stages=-1,
                 with_csp=False,        # only for yolov4.
                 dropblock_stage=None,  # dropblock
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
                 norm_eval=True):

        super(Darknet, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for darknet')
        if dropblock_stage is not None and \
                not set(dropblock_stage).issubset(self.dropblock_stage):
            raise KeyError(f'invalid dropblock_stage {dropblock_stage} for darknet')

        self.depth = depth
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.layers, self.channels = self.arch_settings[depth]
        self.drop_stage = [False] * len(self.dropblock_stage) if \
                          dropblock_stage is None else [x in dropblock_stage
                          for x in self.dropblock_stage]

        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        # conv_1x1
        self.conv1 = ConvModule(3, 32, 3, padding=1, **cfg)  # init_weights() is in ConvModule.
        self.cr_blocks = ['conv1']

        # replicate the Res or CSP module block.
        for i, n_layers in enumerate(self.layers):
            layer_name = f'conv_res_block{i + 1}'
            in_c, out_c = self.channels[i]
            # yolov4: Csp
            if with_csp:
                conv_module = CspResBlock(in_c, out_c, n_layers,
                                          is_first_block=(i == 0),
                                          is_dropblock=self.drop_stage[i],
                                          **cfg)
            # yolov3: Res
            else:
                conv_module = self.make_conv_res_block(in_c, out_c, n_layers,
                                                       is_dropblock=self.drop_stage[i],
                                                       **cfg)

            self.add_module(layer_name, conv_module)
            self.cr_blocks.append(layer_name)

        self.norm_eval = norm_eval

    def forward(self, x):
        outs = []
        for i, layer_name in enumerate(self.cr_blocks):
            cr_block = getattr(self, layer_name)
            x = cr_block(x)
            if i in self.out_indices:
                outs.append(x)

        return tuple(outs)  # the bigger features map are first same as resnet.

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
        super(Darknet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    @staticmethod
    def make_conv_res_block(in_channels,
                            out_channels,
                            res_repeat,
                            is_dropblock=False,
                            conv_cfg=None,
                            norm_cfg=dict(type='BN', requires_grad=True),
                            act_cfg=dict(type='LeakyReLU', negative_slope=0.1)):
        """In Darknet backbone, ConvLayer is usually followed by ResBlock. This
        function will make that. The Conv layers always have 3x3 filters with
        stride=2. The number of the filters in Conv layer is the same as the
        out channels of the ResBlock.

        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            res_repeat (int): The number of ResBlocks.
            is_dropblock(bool): wether dropblock or not.
            conv_cfg (dict): Config dict for convolution layer. Default: None.
            norm_cfg (dict): Dictionary to construct and config norm layer.
                Default: dict(type='BN', requires_grad=True)
            act_cfg (dict): Config dict for activation layer.
                Default: dict(type='LeakyReLU', negative_slope=0.1).
        """

        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        model = nn.Sequential()
        # conv, 32, 64, 3x3, stride=2, pad=1
        model.add_module('conv',
                         ConvModule(in_channels, out_channels, 3,
                                    stride=2, padding=1, **cfg))
        # repeat the resblock which contains a con_1x1 and conv_3x3 module.
        for idx in range(res_repeat):
            model.add_module('res{}'.format(idx),
                             ResBlock(out_channels, is_dropblock=is_dropblock, **cfg))
        return model


class CspResBlock(nn.Module):
    """
    CspResBlock is illustrated as follow below:
    CSPX = -> CBM(pre) -+->CBM(main) -> ResUnit_X -> CBM(post) -+-> concat -> CBM(final) ->
                        |                                       |
                        +---------------> CBM(short) -----------+
    This class makes the conv_res_block in YOLO v4. It has CSP integrated,
    hence different from the regular conv_res_block build with
    `make_conv_res_block`.
    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        res_repeat (int): The number of ResBlocks.
        is_first_block (bool): Whether the CspResBlock is the
            first in the Darknet. This affects the structure of the
            block. Default: False,
        is_dropblock(bool): wether dropblock in the net.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN', requires_grad=True)
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='LeakyReLU', negative_slope=0.1).
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 res_repeat,
                 is_first_block=False,
                 is_dropblock=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1)):

        super(CspResBlock, self).__init__()

        self.is_dropblock = is_dropblock
        if is_dropblock:
            self.dropblock = DropBlock2D_Pool()

        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        bottleneck_channels = out_channels if is_first_block else in_channels
        self.preconv = ConvModule(
            in_channels, out_channels, 3, stride=2, padding=1, **cfg)
        self.shortconv = ConvModule(
            out_channels, bottleneck_channels, 1, stride=1, **cfg)
        self.mainconv = ConvModule(
            out_channels, bottleneck_channels, 1, stride=1, **cfg)

        self.blocks = nn.Sequential()
        for idx in range(res_repeat):
            if is_first_block:
                self.blocks.add_module('res{}'.format(idx),
                                       ResBlock(bottleneck_channels,
                                                is_dropblock=False, **cfg))
            else:
                self.blocks.add_module('res{}'.format(idx),
                                       ResBlock(bottleneck_channels,
                                                is_dropblock=is_dropblock,
                                                yolo_version='v4', **cfg))

        self.postconv = ConvModule(
            bottleneck_channels, bottleneck_channels, 1, stride=1, **cfg)
        self.finalconv = ConvModule(
            2 * bottleneck_channels, out_channels, 1, stride=1, **cfg)

    def forward(self, x):
        x = self.preconv(x)
        x_short = self.shortconv(x)
        x_main = self.mainconv(x)
        x_main = self.blocks(x_main)
        x_main = self.postconv(x_main)
        x_final = torch.cat((x_main, x_short), 1)
        x_final = self.finalconv(x_final)
        return x_final
