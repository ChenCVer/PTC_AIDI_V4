# -*- coding:utf-8 -*-
import os
import cv2
import copy
import torch
import collections
import numpy as np
from ..hook import HOOKS, Hook

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    raise ImportError(
        'Please run "pip install future tensorboard" to install '
        'the dependencies to use torch.utils.tensorboard '
        '(applicable to PyTorch 1.1 or higher)')
import torchvision.utils as vutils

__all__ = ['DetTensorboardHook']


@HOOKS.register_module()
class DetTensorboardHook(Hook):

    def __init__(self, interval=None):
        self._print_interval_iter = interval
        self._train_summarywriter = None
        self._val_summarywriter = None

    def after_train_iter(self, runner):
        if self._print_interval_iter is None:
            self._print_interval_iter = runner.print_interval_iter
        if self._train_summarywriter is None:
            self._train_summarywriter = SummaryWriter(os.path.join(runner.work_dir, 'train'))
        self._train_summarywriter.add_scalar('train_lr', runner.optimizer.param_groups[0]['lr'],
                                             runner._max_iters)
        self._summary(runner, 'train')

    def after_val_iter(self, runner):
        if self._print_interval_iter is None:
            self._print_interval_iter = runner.print_interval_iter
        if self._val_summarywriter is None:
            self._val_summarywriter = SummaryWriter(os.path.join(runner.work_dir, 'val'))
        self._summary(runner, 'val')

    def _summary(self, runner, mode):
        if mode == 'train':
            summary_fun = self._train_summarywriter
            iters = runner._iter
        else:
            summary_fun = self._val_summarywriter
            iters = runner._val_iter

        if self.every_n_inner_iters(runner, self._print_interval_iter):
            # 主要是记录loss, acc
            results = runner.outputs["log_vars"]
            for name, val in results.items():
                if isinstance(val, torch.Tensor) or isinstance(val, float):
                    summary_fun.add_scalar(name, val, iters)

            # 下面主要是记录gt_bboxes 和 predict_bbox
            img_with_result_bbox_list = []
            img_with_gt_list = []
            with torch.no_grad():
                for idx in range(runner.data_batch["img"].data[0].shape[0]):
                    data_batch_copy = copy.deepcopy(runner.data_batch)
                    summary_img = data_batch_copy["img"].data[0][idx: idx + 1, :, :, :].cuda()
                    summary_img_meta = data_batch_copy["img_metas"].data[0][idx]
                    summary_gt_bboxes = data_batch_copy["gt_bboxes"].data[0][idx].detach().cpu().numpy()
                    summary_gt_labels = data_batch_copy["gt_labels"].data[0][idx].detach().cpu().numpy()
                    # note: middle_result 获取预测结果.
                    middle_result = runner.model.module.simple_test(summary_img, [summary_img_meta])
                    meta = collections.Container
                    orig_data = [summary_img_meta]
                    meta.data = [orig_data]
                    new_img_meta = [meta]
                    data = {"img": [summary_img], "img_metas": new_img_meta}
                    img_with_result_bboxes = runner.model.module.add_result(data, middle_result)
                    img_with_result_bboxes = cv2.resize(img_with_result_bboxes, (512, 512))
                    img_with_result_bbox_list.append(img_with_result_bboxes)

                    img_with_gt_bboxes = runner.model.module.add_gt(data, summary_gt_bboxes, summary_gt_labels)
                    img_with_gt_bboxes = cv2.resize(img_with_gt_bboxes, (512, 512))
                    img_with_gt_list.append(img_with_gt_bboxes)

                data_tensor = torch.from_numpy(np.array(np.stack(img_with_result_bbox_list), np.float))
                gt_tensor = torch.from_numpy(np.array(np.stack(img_with_gt_list), np.float))
                data_tensor = data_tensor.permute(0, 3, 1, 2)
                gt_tensor = gt_tensor.permute(0, 3, 1, 2)
                if mode == "train":
                    train_image_with_bbox = vutils.make_grid(data_tensor, nrow=4, normalize=True, scale_each=True)
                    summary_fun.add_image('train_image_with_prediction', train_image_with_bbox, iters)
                    train_image_with_gt = vutils.make_grid(gt_tensor, nrow=4, normalize=True, scale_each=True)
                    summary_fun.add_image('train_image_with_gt', train_image_with_gt, iters)

                else:
                    val_image_with_bbox = vutils.make_grid(data_tensor, nrow=4, normalize=True, scale_each=True)
                    summary_fun.add_image('val_image_with_prediction', val_image_with_bbox, iters)
                    val_image_with_gt = vutils.make_grid(gt_tensor, nrow=4, normalize=True, scale_each=True)
                    summary_fun.add_image('val_image_with_gt', val_image_with_gt, iters)
