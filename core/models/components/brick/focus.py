import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

__all__ = ["Focus", "Concat"]


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self,
                 in_channels=3,
                 out_channels=80,
                 kernel_size=3,
                 stride=1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1)):
        super(Focus, self).__init__()
        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        # build conv where
        self.focus = ConvModule(in_channels * 4, out_channels, kernel_size,
                                stride=stride, padding=kernel_size // 2, **cfg)

    def forward(self, x):
        """
        function: x(b,c,w,h) -> y(b,4c,w/2,h/2)
        """
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2],
                       x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        x = self.focus(x)
        return x


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)
