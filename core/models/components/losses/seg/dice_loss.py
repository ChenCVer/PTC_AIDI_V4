# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.models.builder import LOSSES


__all__ = [
    'BinaryDiceLoss',
    'MultiDiceLoss'

]


@LOSSES.register_module()
class BinaryDiceLoss(nn.Module):
    """
    Dice loss of binary class
    """

    def __init__(self, smooth=1e-6, p=2, reduction='mean', loss_weight=1.0):

        """
        :param smooth: A float number to smooth loss, and avoid NaN error, default: 1.
        :param p: Denominator value: \\sum{x^p} + \\sum{y^p}, default: 2.
        :param reduction: Reduction method to apply, return mean over batch if 'mean',
                          return sum if 'sum', return a tensor of shape [N,] if 'none'.
        """
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, predictive, target):
        """

        :param predictive: A tensor of shape [N, *].
        :param target:  A tensor of shape same with predict.
        :return: Loss tensor according to arg reduction

        Raise:
        Exception if unexpected reduction
        """
        assert predictive.shape[0] == target.shape[0], "predict & target batch size don't match"
        predictive = F.sigmoid(predictive)
        predictive = predictive.contiguous().view(-1, 1)
        target = target.contiguous().view(-1, 1).float()

        num = 2 * torch.sum(predictive * target) + self.smooth
        den = torch.sum(predictive) + torch.sum(target) + self.smooth
        loss = 1 - num / den

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

        return self.loss_weight * loss


@LOSSES.register_module()
class MultiDiceLoss(nn.Module):
    """
    Dice loss of binary class.
    """

    def __init__(self, smooth=1e-6, p=2, reduction='mean'):

        """
        :param smooth: A float number to smooth loss, and avoid NaN error, default: 1.
        :param p: Denominator value: \\sum{x^p} + \\sum{y^p}, default: 2.
        :param reduction: Reduction method to apply, return mean over batch if 'mean',
                          return sum if 'sum', return a tensor of shape [N,] if 'none'
        """

        super(MultiDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predictive, target):
        """
        # TODO: 由于背景像素太多, 综合考虑到这里for i in range(predictive.shape[1] - 1)主要是不计算背景类别的DiceLoss损失
        # TODO: 后续需要考虑是否需要计算背景类别的DiceLoss.
        :param predictive: A tensor of shape [N, *]
        :param target: A tensor of shape same with predict
        :return: Loss tensor according to arg reduction

        Raise:
        Exception if unexpected reduction

        """
        assert predictive.shape[0] == target.shape[0], "predict & target batch size don't match"
        predictive = F.softmax(predictive)
        all_loss = 0
        for i in range(1, predictive.shape[1]):  # TODO: 背景类不计算DiceLoss, 背景通道在第0通道
            predict_one = predictive[:, i, :, :]
            target_one = target[:, i, :, :]
            predict_one = predict_one.contiguous().view(-1, 1)
            target_one = target_one.contiguous().view(-1, 1).float()

            num = 2 * torch.sum(predict_one * target_one) + self.smooth
            den = torch.sum(predict_one) + torch.sum(target_one) + self.smooth
            loss = 1 - num / den
            all_loss += loss

        loss = all_loss / predictive.shape[1]

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))