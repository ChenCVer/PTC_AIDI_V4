import torch
import torch.nn as nn
import torch.nn.functional as F
from .registry import ACTIVATION_LAYERS


# yolov4专用
# TODO: 有cuda加速实现, 后期看看cuda的加速代码.
@ACTIVATION_LAYERS.register_module()
class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))