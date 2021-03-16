# -*- coding:utf-8 -*-
import os
import gc
import cv2
import copy
import torch
import numpy as np
from ..hook import HOOKS, Hook
import torch.nn.functional as F
from collections import OrderedDict
from rraitools.visualtools.feature_vistools import ClsGradClassActivationMappingVis
from sklearn.metrics import precision_score, recall_score, f1_score, \
    confusion_matrix, accuracy_score

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    raise ImportError(
        'Please run "pip install future tensorboard" to install '
        'the dependencies to use torch.utils.tensorboard '
        '(applicable to PyTorch 1.1 or higher)')
import torchvision.utils as vutils

__all__ = ['ClsTensorboardHook']


@HOOKS.register_module()
class ClsTensorboardHook(Hook):
    def __init__(self, show_cam=False, interval=None):

        self.show_cam = show_cam
        self._print_interval_iter = interval
        self._train_summarywriter = None

    def after_train_iter(self, runner):
        if self._print_interval_iter is None:
            self._print_interval_iter = runner.print_interval_iter

        if self._train_summarywriter is None:
            self._train_summarywriter = SummaryWriter(os.path.join(runner.work_dir, 'train'))
        self._train_summarywriter.add_scalar('train_lr', runner.optimizer.param_groups[0]['lr'], runner._iter)

        if self.every_n_inner_iters(runner, self._print_interval_iter):
            # loss
            for name, val in runner.outputs.items():
                if isinstance(val, torch.Tensor) or isinstance(val, float):
                    self._train_summarywriter.add_scalar(name, val, runner._iter)
                elif isinstance(val, OrderedDict):
                    for key, value in val.items():
                        if isinstance(value, float):
                            self._train_summarywriter.add_scalar(key, value, runner._iter)

            # image
            input_tensor = runner.data_batch["img"].data[0]
            if input_tensor.shape[1] == 6:  # 分类含有参考图的情况
                # 这种可视化方法，在有些电脑上会显示不同步
                # w方向拼接可视化
                image = torch.cat([input_tensor[:, 0:3, ...], input_tensor[:, 3:, ...]], dim=-1)
                train_image = vutils.make_grid(image, normalize=True, scale_each=True)
                self._train_summarywriter.add_image('train_image', train_image, runner._iter)

            else:
                train_image = vutils.make_grid(input_tensor, normalize=True, scale_each=True)
                self._train_summarywriter.add_image('train_image', train_image, runner._iter)

                if self.show_cam:
                    # Grad-cam visualization
                    # TODO: 可视化grad-cam太费显存(随时间越来越多), 且训练时间明显变长, 慎用!
                    cam_img_list = []
                    # 这里必须将model变成eval模式.
                    model = copy.deepcopy(runner.model)
                    grid_cam = ClsGradClassActivationMappingVis(model.eval())
                    # 这里需要获得每张图像的热力图, 然后进行拼装组合
                    for idx in range(runner.data_batch["img"].data[0].shape[0]):
                        grid_cam.set_hook_style(3, runner.data_batch["img"].data[0][idx:idx+1].shape)
                        cam_img = grid_cam.run(runner.data_batch["img"].data[0][idx:idx+1])["cam_img"]
                        cam_img = cv2.cvtColor(cam_img, cv2.COLOR_BGR2RGB)  # 这个必须要加, 不加会出问题.
                        cam_img_list.append(cam_img / 255.0)

                    cam_tensor = torch.from_numpy(np.array(cam_img_list)).permute([0, 3, 1, 2])
                    cam_image = vutils.make_grid(cam_tensor.float(), normalize=True, scale_each=True)
                    self._train_summarywriter.add_image('train_image_cam', cam_image, runner._iter)
                    runner.model.train()
                    # 手动内存释放
                    del grid_cam
                    torch.cuda.empty_cache()
                    gc.collect()

    # def after_val_epoch(self, runner):
    #     targets = [value["cls_id"] for value in runner.data_loader.dataset.data_infos]
    #     results = runner.log_buffer.val_history["prediction"]
    #     val_cfg = runner.cfg.get('evaluation', {})
    #     thres_score = val_cfg["thres_score"]
    #     results = [x[0] for x in results]
    #     results_cat = torch.cat(results, dim=0)
    #     softmax_results = F.softmax(results_cat, dim=1)
    #     predict = torch.argmax(softmax_results, dim=1)
    #     max_inex_p = torch.max(softmax_results, dim=1)[0]
    #     # 这里考虑到, 虽然其类别都预测正确, 但是其概率值并不高, 这种情况,也是网络没有训练好, 泛化能力欠佳.
    #     # 这里,假设predict是预测正确, 但是他的概率值没达到预设值, 则变成相反类别.
    #     if thres_score is not None:
    #         predict = torch.where(max_inex_p > thres_score, predict, max(targets) - predict).tolist()
    #     # 以上过程是获取label和predict的过程, 以下过程计算统计指标
    #     eval_metrics = {}
    #     eval_metrics["val_confusion_matrix"] = str(confusion_matrix(targets, predict))
    #     eval_metrics["val_accuracy"] = accuracy_score(targets, predict)  # accuracy
    #     if max(targets) <= 1:
    #         eval_metrics["val_precison"] = precision_score(targets, predict)  # precision
    #         eval_metrics["val_recall"] = recall_score(targets, predict)  # recall
    #         eval_metrics["val_f1-score"] = f1_score(targets, predict)  # f1-score
    #
    #     for name, val in eval_metrics.items():
    #         if isinstance(val, torch.Tensor) or isinstance(val, float):
    #             self._train_summarywriter.add_scalar(name, val, runner._val_epoch)