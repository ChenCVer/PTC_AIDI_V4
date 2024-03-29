import torch
import torch.nn as nn
from core.models.builder import NECKS
from mmcv.cnn import caffe2_xavier_init

__all__ = [
    "GlobalAveragePooling",
]


@NECKS.register_module()
class GlobalAveragePooling(nn.Module):
    """Global Average Pooling neck.

    Note that we use `view` to remove extra channel after pooling.
    We do not use `squeeze` as it will also remove the batch dimension
    when the tensor has a batch dimension of size 1, which can lead to
    unexpected errors.
    """

    def __init__(self):
        super(GlobalAveragePooling, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def init_weights(self):
        """Initialize the weights of module"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                caffe2_xavier_init(m)

    def forward(self, inputs):
        if isinstance(inputs, tuple):
            outs = tuple([self.gap(x) for x in inputs])
            outs = tuple([out.view(x.size(0), -1) for out, x in zip(outs, inputs)])
        elif isinstance(inputs, torch.Tensor):
            outs = self.gap(inputs)
            outs = outs.view(inputs.size(0), -1)
        else:
            raise TypeError('neck inputs should be tuple or torch.tensor')
        return outs
