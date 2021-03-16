# Copyright (c) 2019 Western Digital Corporation or its affiliates.

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from core.models.registry import NECKS


__all__ = ['YoloNeck',
           ]


class AttentionSAM(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='Sigmoid')):
        super(AttentionSAM, self).__init__()

        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.attention_sam = ConvModule(in_channels, out_channels, kernel_size, **cfg)

    def forward(self, x):
        return self.attention_sam(x) * x


class DetectionBlock(nn.Module):
    """Detection block in YOLO neck.
    Note: 相当于5个CBL [+ SPP]结构
    Let out_channels = n, the DetectionBlock normally contains 5 ConvModules,
    Their sizes are 1x1xn, 3x3x2n, 1x1xn, 3x3x2n, and 1x1xn respectively.
    If the spp is on, the DetectionBlock contains 6 ConvModules and
    3 pooling layers, sizes are 1x1xn, 3x3x2n, 1x1xn,
    5x5 maxpool, 9x9 maxpool, 13x13 maxpool, 1x1xn, 3x3x2n, 1x1xn.
    The input channel is arbitrary (in_channels)

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        spp_scales (tuple | None): A set of sizes for spatial pyramid pooling.
            When set to None, the spp is disabled. Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN', requires_grad=True).
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='LeakyReLU', negative_slope=0.1).
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 attention_sam=False,  # yolov4
                 spp_scales=None,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1)):
        super(DetectionBlock, self).__init__()
        self.spp_on = spp_scales is not None
        self.attention_sam = attention_sam
        double_out_channels = out_channels * 2

        # shortcut
        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv1 = ConvModule(in_channels, out_channels, 1, **cfg)
        self.conv2 = ConvModule(out_channels, double_out_channels, 3, padding=1, **cfg)
        self.conv3 = ConvModule(double_out_channels, out_channels, 1, **cfg)

        # spp
        if self.spp_on:
            self.poolers = [
                nn.MaxPool2d(size, 1, padding=(size - 1) // 2)
                for size in spp_scales
            ]
            self.conv_spp = ConvModule(out_channels * (len(spp_scales) + 1),
                                       out_channels, 1, **cfg)

        # conv
        self.conv4 = ConvModule(out_channels, double_out_channels, 3, padding=1, **cfg)
        # sam
        if self.attention_sam:
            self.sam = AttentionSAM(double_out_channels, double_out_channels, 1)
        # conv
        self.conv5 = ConvModule(double_out_channels, out_channels, 1, **cfg)

    def forward(self, x):
        tmp = self.conv1(x)
        tmp = self.conv2(tmp)
        tmp = self.conv3(tmp)

        if self.spp_on:
            spp_feats = [tmp] + [pooler(tmp) for pooler in self.poolers]
            tmp = torch.cat(spp_feats[::-1], 1)
            tmp = self.conv_spp(tmp)

        tmp = self.conv4(tmp)
        if self.attention_sam:
            tmp = self.sam(tmp)
        out = self.conv5(tmp)
        return out


@NECKS.register_module()
class YoloNeck(nn.Module):
    """The neck of YOLOV3.

    It can be treated as a simplified version of FPN. It
    will take the result from Darknet backbone and do some upsampling and
    concatenation. It will finally output the detection result.

    Note:
        The input feats should be from top to bottom.
            i.e., from high-lvl to low-lvl
        But YOLOV3Neck will process them in reversed order.
            i.e., from bottom (high-lvl) to top (low-lvl)

    Args:
        num_scales (int): The number of scales / stages.
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        spp_scales (tuple): A set of sizes for spatial pyramid pooling.
            When set to None, the spp is disabled. Default: None.
        yolo_version (str): The version of YOLO to build, must be 'v3' or 'v4',
            Default: 'v3'
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN', requires_grad=True)
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='LeakyReLU', negative_slope=0.1).
    """

    def __init__(self,
                 num_scales,
                 in_channels,
                 out_channels,
                 spp_scales=None,
                 yolo_version='v3',
                 attention_sam=False,  # only for yolov4
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1)):
        super(YoloNeck, self).__init__()
        if yolo_version not in ('v3', 'v4'):
            raise NotImplementedError('Only YOLO v3 and v4 are supported.')
        assert (num_scales == len(in_channels) == len(out_channels))
        self.num_scales = num_scales
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spp_on = spp_scales is not None
        self.yolo_version = yolo_version

        # shortcut
        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        # If spp is enabled, the first DetectionBlock is built with a spp
        # module inserted into it and the other DetectionBlock stays unchanged.
        self.detect1 = DetectionBlock(self.in_channels[0],
                                      self.out_channels[0],
                                      spp_scales=spp_scales, **cfg)

        # To support arbitrary scales, the code looks awful, but it works.
        # Better solution is welcomed.
        if yolo_version == "v3":
            for i in range(1, self.num_scales):
                in_c, out_c = self.in_channels[i], self.out_channels[i]
                self.add_module(f'conv{i}', ConvModule(in_c, out_c, 1, **cfg))
                # in_c + out_c : High-lvl feats will be cat with low-lvl feats
                self.add_module(f'detect{i + 1}', DetectionBlock(in_c + out_c, out_c, **cfg))
        else:  # v4
            for i in range(1, self.num_scales):
                in_c, out_c = self.in_channels[i], self.out_channels[i]
                self.add_module(f'upsample_conv{i}', ConvModule(in_c, out_c, 1, **cfg))
                self.add_module(f'feat_conv{i}', ConvModule(in_c, out_c, 1, **cfg))
                if i == self.num_scales - 1:  # 这里要创建sam, 因为i=self.num_scales-1时, 此时是downsample.
                    self.add_module(f'detect{i + 1}', DetectionBlock(in_c, out_c, attention_sam, **cfg))
                else:
                    self.add_module(f'detect{i + 1}', DetectionBlock(in_c, out_c, **cfg))

            # downsampling PANet path, SAM is in downsample.
            for i in range(self.num_scales, self.num_scales * 2 - 1):
                ds_channel_idx = self.num_scales * 2 - 1 - i
                ds_in_c = self.out_channels[ds_channel_idx]
                ds_out_c = self.in_channels[ds_channel_idx]
                det_channel_idx = ds_channel_idx - 1
                det_in_c = self.in_channels[det_channel_idx]
                det_out_c = self.out_channels[det_channel_idx]
                self.add_module(f'downsample_conv{i - self.num_scales + 1}',
                                ConvModule(ds_in_c, ds_out_c, 3, stride=2, padding=1, **cfg))
                self.add_module(f'detect{i + 1}', DetectionBlock(det_in_c, det_out_c,
                                                                 attention_sam, **cfg))

    def forward(self, feats):

        assert len(feats) == self.num_scales
        outs = []
        # P5->O5
        out = self.detect1(feats[-1])
        outs.append(out)

        if self.yolo_version == 'v3':
            for i, x in enumerate(reversed(feats[:-1])):
                conv = getattr(self, f'conv{i + 1}')
                # Channel conversion firstly
                tmp = conv(out)

                # umsampling with 2 times, and then concat.
                tmp = F.interpolate(tmp, scale_factor=2)
                tmp = torch.cat((tmp, x), 1)

                detect = getattr(self, f'detect{i + 2}')
                # 5 times CBL
                out = detect(tmp)
                outs.append(out)
        else:  # v4
            # FPN
            for i, x in enumerate(reversed(feats[:-1])):
                upsample_conv = getattr(self, f'upsample_conv{i + 1}')
                tmp = upsample_conv(out)

                feat_conv = getattr(self, f'feat_conv{i + 1}')
                tmp_x = feat_conv(x)

                # Cat with low-lvl feats
                tmp = F.interpolate(tmp, scale_factor=2)
                tmp = torch.cat((tmp_x, tmp), 1)

                detect = getattr(self, f'detect{i + 2}')
                out = detect(tmp)
                outs.append(out)

            cur_feat = outs[-1]
            # PAN
            for i in range(self.num_scales - 1):
                downsample_conv = getattr(self, f'downsample_conv{i + 1}')
                tmp = downsample_conv(cur_feat)
                tmp = torch.cat((tmp, outs[-2 - i]), 1)

                detect = getattr(self, f'detect{i + self.num_scales + 1}')
                cur_feat = detect(tmp)
                outs[-2 - i] = cur_feat

        return tuple(outs)

    def init_weights(self):
        """Initialize the weights of module."""
        # init is done in ConvModule
        pass
