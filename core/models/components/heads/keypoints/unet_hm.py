# -*- coding:utf-8 -*-
import cv2
import torch
import numpy as np
import torch.nn as nn
from mmcv import imresize
from ..base import BaseHead
from core.models.registry import HEADS
from core.models.builder import build_loss
from ...brick.unet_parts import UpSample, OutConv

__all__ = ['UnetHead_Hm']


@HEADS.register_module()
class UnetHead_Hm(BaseHead):
    def __init__(self,
                 num_classes,
                 root_channels=16,
                 layer_num=4,
                 kernel_size=3,
                 use_double_conv=True,
                 shortcut=False,
                 norm=nn.BatchNorm2d,
                 loss=None,
                 train_cfg=None,
                 valid_cfg=None,
                 test_cfg=None):

        super(UnetHead_Hm, self).__init__()

        self.train_cfg = train_cfg
        self.valid_cfg = valid_cfg
        self.test_cfg = test_cfg

        # build layers
        upsample_list = []
        for i in range(layer_num):
            upsample_list.append(UpSample(root_channels * 3 * (2 ** i),
                                          root_channels * (2 ** i),
                                          kernel_size, use_double_conv,
                                          norm, shortcut))
        self.up_convs = nn.ModuleList(upsample_list)
        self.outc = OutConv(root_channels, num_classes)

        # 用于nms postprocess
        self.maxpool_3x3 = torch.nn.MaxPool2d([3, 3], [1, 1], [1, 1], [1, 1])
        self.maxpool_5x5 = torch.nn.MaxPool2d([5, 5], [1, 1], [2, 2], [1, 1])
        self.maxpool_7x7 = torch.nn.MaxPool2d([7, 7], [1, 1], [3, 3], [1, 1])
        self.maxpool_9x9 = torch.nn.MaxPool2d([9, 9], [1, 1], [4, 4], [1, 1])

        # build losses
        self.losses = build_loss(loss)

    def init_weights(self):
        """
        Note:
            segmentation training generally does not require (rarely required) pre-training
            weights, especially in the industrial field.
        """
        pass

    def forward(self, input_tensor):
        x = input_tensor[-1]
        for i in range(len(self.up_convs), 1, -1):
            x = self.up_convs[i - 1](x, input_tensor[i - 1])
        x = self.up_convs[0](x, input_tensor[0])
        x = self.outc(x)

        return x

    def forward_train(self, x, img_metas, gt_semantic_seg,
                      target_ignore=None, **kwargs):
        """
        TODO: 这里重写了BaseHead类的forward_train方法, 如果后续其他分割网络也有head,
         则又需要在其他head中重写一遍forward_train方法. 因此, 后续考虑和BaseHead中
         的forward_train方法统合并.
        """
        # head
        outs = self(x)
        losses = self.loss(outs, gt_semantic_seg, img_metas,
                           target_ignore=target_ignore)

        return losses

    def loss(self, pred, gt_semantic_seg,
             img_metas, target_ignore=None):
        losses = self.losses.compute_loss(pred, gt_semantic_seg, img_metas)
        return losses

    def get_results(self,
                    result,
                    img_metas,
                    cfg=None,
                    rescale=False,
                    **kwargs):
        """
        Args:
            result: predictions from net (outs doesn't flow the softmax or sigmoid layer.)
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used
            rescale (bool): If True, return boxes in original image space

        Returns: predict mask

        """
        if result.shape[1] == 1:
            # sigmoid模式
            output = torch.sigmoid(result)
            # centernet中用maxpool替代nms操作
            hmax0 = torch.max_pool2d(output, [3, 3], [1, 1], [1, 1], [1, 1], False)
            hmax1 = torch.max_pool2d(output, [5, 5], [1, 1], [2, 2], [1, 1], False)
            hmax2 = torch.max_pool2d(output, [7, 7], [1, 1], [3, 3], [1, 1], False)
            hmax3 = torch.max_pool2d(output, [9, 9], [1, 1], [4, 4], [1, 1], False)
            hmax = torch.max(torch.cat([hmax0, hmax1, hmax2, hmax3], 1))
            keep = (hmax == result).float().to(device=result.device)
            output = torch.mul(output, keep)

            score_thr = cfg.get('score_thr', None)
            if score_thr is not None:
                predictive = np.zeros_like(output, dtype=np.uint8)
                predictive[output > score_thr] = 255
            else:
                predictive = np.uint8(output[..., 0] * 255.0)
            predictive = cv2.cvtColor(predictive, cv2.COLOR_GRAY2RGB)
        else:
            # softmax模式, softmax模式也可能需要score_thr, 如果没有超过阈值,
            # 直接变为背景, 0通道默认是背景.
            output = torch.softmax(result, dim=1)
            prediction = torch.argmax(output, dim=1)
            max_inex_p = torch.max(output, dim=1)[0]
            score_thr = cfg.get('score_thr', None)
            if score_thr is not None:
                prediction[max_inex_p < score_thr] = 0  # 阈值操作.
            prediction = prediction.detach().cpu().numpy().squeeze()
            # todo: 2021-01-06: 多分割可能存在问题.
            class_list = list(cfg.get("class_order_dict").values())
            predictive = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)
            for idx, color in enumerate(class_list):
                predictive[prediction == idx] = color

        if rescale:
            ori_shape = img_metas[0]["ori_shape"]
            if ori_shape[:2] != predictive.shape[:2]:
                predictive = imresize(predictive, ori_shape[:2][::-1])

        return predictive