# -*- coding:utf-8 -*-
from rraitools import decode_onehot_from_tensor
from ..hook import HOOKS, Hook
import torch
import os


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    raise ImportError(
        'Please run "pip install future tensorboard" to install '
        'the dependencies to use torch.utils.tensorboard '
        '(applicable to PyTorch 1.1 or higher)')
import torchvision.utils as vutils

__all__ = ['SegTensorboardHook']


@HOOKS.register_module()
class SegTensorboardHook(Hook):
    def __init__(self,
                 num_class=1,
                 color_list=None,
                 interval=None):

        self._num_class = num_class
        self._color_list = color_list
        self._print_interval_iter = interval
        self._train_summarywriter = None
        self._val_summarywriter = None

    def after_train_iter(self, runner):
        if self._print_interval_iter is None:
            # TODO: print_interval_iter is not exist, modify!
            self._print_interval_iter = runner.print_interval_iter
        if self._train_summarywriter is None:
            self._train_summarywriter = SummaryWriter(os.path.join(runner.work_dir, 'train'))
        self._train_summarywriter.add_scalar('train_lr', runner.optimizer.param_groups[0]['lr'],
                                             runner._iter)
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
            # loss and predict
            results = runner.outputs["log_vars"]
            for name, val in results.items():
                if isinstance(val, torch.Tensor) or isinstance(val, float):
                    summary_fun.add_scalar(name, val, iters)

                # tensorboard prediction tensor
                elif isinstance(val, list):
                    for index, data in enumerate(val):
                        if len(data.shape) == 4:
                            data = decode_onehot_from_tensor(data, self._color_list, self._num_class, True, model='bgr')
                            # 在多分类模式下，如果normalize=False, scale_each=False,显示会很奇怪
                            train_predict = vutils.make_grid(data.double(), normalize=True, scale_each=True, padding=0)
                            summary_fun.add_image('{}_predict_{}'.format(mode, index), train_predict, iters)
            # image
            input_tensor = runner.data_batch["img"].data[0]
            if input_tensor.shape[1] == 6:  # 带有参考图通道
                # 这种可视化方法，在有些电脑上会显示不同步
                # w方向拼接可视化
                image = torch.cat([input_tensor[:, 0:3, ...], input_tensor[:, 3:, ...]], dim=-1)
                train_image = vutils.make_grid(image, normalize=True, scale_each=True)
                summary_fun.add_image('{}_image'.format(mode), train_image, iters)
            else:
                train_image = vutils.make_grid(input_tensor, normalize=True, scale_each=True)
                summary_fun.add_image('{}_image'.format(mode), train_image, iters)

            # mask(label)
            target = runner.data_batch["gt_semantic_seg"].data[0]
            if not isinstance(target, list):
                target = [target]
            for index, data in enumerate(target):
                data = decode_onehot_from_tensor(data.double(), self._color_list, self._num_class, False, model='bgr')
                train_target = vutils.make_grid(data.double(), normalize=True, scale_each=True, padding=0)  # B C H W
                summary_fun.add_image('{}_target_{}'.format(mode, index), train_target, iters)