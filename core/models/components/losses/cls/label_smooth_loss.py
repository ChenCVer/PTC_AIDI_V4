import torch
import torch.nn as nn
import torch.nn.functional as F
from core.models.builder import LOSSES
from core.models.components.losses.utils import weight_reduce_loss

__all__ = [
    "LabelSmoothLoss",
]


def label_smooth(pred,
                 label,
                 label_smooth_val,
                 avg_smooth_val,
                 temperature=1.0,
                 weight=None,
                 reduction='mean',
                 avg_factor=None):
    # # element-wise losses
    one_hot = torch.zeros_like(pred)
    one_hot.fill_(avg_smooth_val)
    label = label.view(-1, 1)
    one_hot.scatter_(1, label, 1 - label_smooth_val + avg_smooth_val)
    loss = -torch.sum(F.log_softmax(pred / temperature, 1) * (one_hot.detach()), dim=1)

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


@LOSSES.register_module()
class LabelSmoothLoss(nn.Module):

    def __init__(self,
                 label_smooth_eps,
                 num_classes,
                 temperature=1.0,
                 reduction='mean',
                 loss_weight=1.0):
        super(LabelSmoothLoss, self).__init__()
        self.temperature = temperature
        self.label_smooth_val = label_smooth_eps
        self.avg_smooth_val = self.label_smooth_val / num_classes
        self.reduction = reduction
        self.loss_weight = loss_weight

        self.cls_criterion = label_smooth

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            self.label_smooth_val,
            self.avg_smooth_val,
            self.temperature,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)

        return loss_cls
