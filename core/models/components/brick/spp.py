import torch
from torch import nn
from mmcv.cnn import ConvModule


class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13),
                 conv_cfg=None, norm_cfg=None,
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.cv1 = ConvModule(c1, c_, 1, stride=1, **cfg)
        self.cv2 = ConvModule(c_ * (len(k) + 1), c2, 1, stride=1, **cfg)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))