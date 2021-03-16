# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from mmcv.cnn import normal_init
from core.models.registry import HEADS
from core.models.builder import build_loss
from core.models.components.heads.base import BaseHead


__all__ = ["LinearClsHead",]


@HEADS.register_module()
class LinearClsHead(BaseHead):
    """Linear classifier head: Backbone+Gap+Fc
    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        loss (dict): Config of classification loss.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss=None,
                 train_cfg=None,
                 valid_cfg=None,
                 test_cfg=None):
        super(LinearClsHead, self).__init__()

        self.train_cfg = train_cfg
        self.valid_cfg = valid_cfg
        self.test_cfg = test_cfg

        self.in_channels = in_channels
        self.num_classes = num_classes

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        self._init_layers()

        # build losses
        self.losses = build_loss(loss)

    def _init_layers(self):
        self.fc = nn.Linear(self.in_channels, self.num_classes)

    def init_weights(self):
        normal_init(self.fc, mean=0, std=0.01, bias=0)

    def forward(self, inputs):
        if isinstance(inputs, tuple):
            if len(inputs) == 1:
                outs = self.fc(inputs[0])
            else:
                outs = [self.fc(x) for x in inputs]
        elif isinstance(inputs, torch.Tensor):
            outs = self.fc(inputs)
        else:
            raise TypeError('neck inputs should be tuple or torch.tensor')

        return outs

    def forward_train(self, x, img_metas, label,
                      target_ignore=None, **kwargs):
        # head
        outs = self(x)
        losses = self.loss(outs, label, img_metas,
                           target_ignore=target_ignore)

        return losses

    def loss(self, pred, label,
             img_metas, target_ignore=None):
        label = label.view(-1).long()
        losses = self.losses.compute_loss(pred, label, img_metas)
        return losses

    def get_results(self, outs, *args, **kwargs):
        """
        Args:
            outs: predictions from net (outs doesn't flow the softmax or sigmoid layer.)
        Returns: predict mask
        """
        # TODO: 2021-01-07这里由于grad-cam代码中要求的输入问题, 很难做到像分割, 检测那样对网络
        #  的最后一层进行结果解析. 后续考虑在代码重构的时候和系统学习可视化部分时, 对这部分代码再次进行重写.

        return outs
