# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.models.builder import LOSSES

__all__ = [
    'TverskyLoss',

]


@LOSSES.register_module()
class TverskyLoss(nn.Module):
    """
    参考: https://arxiv.org/abs/1810.07842
    github: https://github.com/nabsabraham/focal-tversky-unet/blob/master/losses.py
    """

    def __init__(self, smooth=1e-6, alpha=0.7, gamma=1.3,
                 reduction='mean', loss_weight=1.0, is_focal=False):
        super(TverskyLoss, self).__init__()
        self.smooth = smooth
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.is_focal = is_focal

    def forward(self, predictive, target):
        assert predictive.shape[0] == target.shape[0], \
            "predict & target batch size don't match"
        predictive = F.sigmoid(predictive)
        predictive = predictive.contiguous().view(-1, 1)
        target = target.contiguous().view(-1, 1).float()

        # true_pos = 2*torch.sum(predict * target) is alse ok
        true_pos = torch.sum(predictive * target) + self.smooth
        false_neg = torch.sum(target * (1 - predictive))
        false_pos = torch.sum((1 - target) * predictive)
        den = true_pos + self.alpha * false_neg + (1 - self.alpha) * false_pos + self.smooth
        pt_1 = true_pos / den

        if self.is_focal:
            loss = torch.pow((1 - pt_1), self.gamma)
        else:
            loss = 1.0 - pt_1

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

        return self.loss_weight * loss
