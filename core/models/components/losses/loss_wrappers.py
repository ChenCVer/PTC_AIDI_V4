# -*- coding:utf-8 -*-
from core.utils import build_from_cfg
from core.models.builder import LOSSES


__all__ = [
    'GeneralLosser',
]


@LOSSES.register_module()
class GeneralLosser(object):
    """
    GeneralLosser主要是为了应对多任务损失监督的场景中, 当然也兼容单任务损失监督场景中.
    """

    def __init__(self, losses):
        assert losses is not None, "losses must be not None"
        self.train_loss_list = self.__parse_lossfun(losses)

    def compute_loss(self, prediction, targets, img_metas, cfg=None):

        outputs_dict = {}
        # todo: 将来要去掉prediction的list限制, 主要考虑到:
        #  core/models/base/base.py中的_parse_losses函数解析兼容问题.
        if not isinstance(prediction, list):
            prediction = [prediction]

        outputs_dict['prediction'] = prediction
        single_loss = []
        for loss_fun in self.train_loss_list:
            single_loss.append(loss_fun(prediction[0], targets))

        for j in range(len(single_loss)):
            loss_name = self.train_loss_list[j].__class__.__name__
            outputs_dict[loss_name] = single_loss[j]

        return outputs_dict

    def __parse_lossfun(self, losses):
        """
        func: 此函数用于解析多任务监督中损失函数
        :param losses: et. <class 'list'>:
        losses=[[dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                 dict(type='RRBinaryDiceLoss', use_sigmoid=False, loss_weight=4.0)], ],
        :return:
        """
        train_losses = []
        for idx, loss_fun in enumerate(losses):
            loss = build_from_cfg(loss_fun, LOSSES)
            train_losses.append(loss)

        return train_losses
