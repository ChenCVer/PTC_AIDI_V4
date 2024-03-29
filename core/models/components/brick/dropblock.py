import torch
import torch.nn.functional as F
from torch import nn

torch.manual_seed(0)


class DropBlock2D_Conv(nn.Module):
    r"""Randomly zeroes spatial blocks of the input tensor.


    As described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.

    Args:
        keep_prob (float, optional): probability of an element to be kept.
        Authors recommend to linearly decrease this value from 1 to desired
        value.
        block_size (int, optional): size of the block. Block size in paper
        usually equals last feature map dimensions.

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)

    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890
    """

    def __init__(self, keep_prob=0.9, block_size=7):
        super(DropBlock2D_Conv, self).__init__()
        assert block_size % 2 == 1, "block_size must be odd!"
        self.keep_prob = keep_prob
        self.block_size = block_size

    def forward(self, input):
        if not self.training or self.keep_prob == 1:
            return input
        gamma = (1. - self.keep_prob) / self.block_size ** 2
        for sh in input.shape[2:]:
            gamma *= sh / (sh - self.block_size + 1)
        M = torch.bernoulli(torch.ones_like(input) * gamma)
        Msum = F.conv2d(M, torch.ones(
            (input.shape[1], 1, self.block_size,
             self.block_size)).to(device=input.device,
                                  dtype=input.dtype),
                        padding=self.block_size // 2,
                        groups=input.shape[1])
        mask = (Msum < 1).to(device=input.device, dtype=input.dtype)
        scale_factor = mask.numel() / mask.sum()
        return input * mask * scale_factor


class DropBlock2D_Pool(nn.Module):
    """
    Notes: 没有参数, 不需要初始化.
    """
    def __init__(self, keep_prob=0.9, block_size=7):
        super(DropBlock2D_Pool, self).__init__()
        assert block_size % 2 == 1, "block_size must be odd!"
        self.keep_prob = keep_prob
        self.block_size = block_size

    def forward(self, input):
        if not self.training or self.keep_prob == 1:
            return input
        gamma = (1. - self.keep_prob) / self.block_size ** 2
        for sh in input.shape[2:]:
            gamma *= sh / (sh - self.block_size + 1)
        M = torch.bernoulli(torch.ones_like(input) * gamma)
        Msum = F.max_pool2d(M, kernel_size=[self.block_size, self.block_size],
                            stride=1, padding=self.block_size // 2)
        mask = (1 - Msum).to(device=input.device, dtype=input.dtype)
        scale_factor = mask.numel() / mask.sum()

        return input * mask * scale_factor