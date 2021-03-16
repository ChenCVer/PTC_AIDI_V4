# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.models.builder import LOSSES

__all__ = [
    "JaccardLoss",
]


@LOSSES.register_module()
class JaccardLoss(nn.Module):
    """
    Dice loss of binary class.
    """

    def __init__(self, smooth=1e-6, p=2, reduction='mean', loss_weight=1.0):
        """
        :param smooth: A float number to smooth loss, and avoid NaN error, default: 1.
        :param p: Denominator value: \\sum{x^p} + \\sum{y^p}, default: 2.
        :param reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'.
        """
        super(JaccardLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, predictive, target):
        """

        :param predictive: A tensor of shape [N, *].
        :param target: A tensor of shape same with predict.
        :return: Loss tensor according to arg reduction.

        Raise:
        Exception if unexpected reduction.

        """
        assert predictive.shape[0] == target.shape[0], "predict & target batch size don't match"
        predictive = F.sigmoid(predictive)
        predictive = predictive.contiguous().view(-1, 1)
        target = target.contiguous().view(-1, 1).float()

        num = torch.sum(predictive * target)
        den = torch.sum(predictive) + torch.sum(target) - num
        # loss = - (num + self.smooth) / (den + self.smooth) is ok,
        # Because they produce the same gradient.
        # you can see the function f(x)=-x and g(x)=1-x, if you want
        # to get hold of the minimum value with gradient descent at
        # the condition of x∈[0, 1]. finally,you can get the same value
        # to f(x) as g(x), namely x is equal to 1.
        loss = 1 - (num + self.smooth) / (den + self.smooth)

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

        return self.loss_weight * loss
