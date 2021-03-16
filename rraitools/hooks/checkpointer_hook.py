import torch
import os
import math
from .hook import Hook
from collections import OrderedDict


def weights_to_cpu(state_dict):
    """Copy a model state_dict to cpu.

    Args:
        state_dict (OrderedDict): Model weights on GPU.

    Returns:
        OrderedDict: Model weights on GPU.
    """
    state_dict_cpu = OrderedDict()
    for key, val in state_dict.items():
        state_dict_cpu[key] = val.cpu()
    return state_dict_cpu


class CheckpointerHook(Hook):
    def __init__(self,
                 save_interval_epoch=5,
                 last_n_epoch=3,
                 model_name='model.pth', cfg=None):

        self.save_interval_epoch = save_interval_epoch
        self.last_n_epoch = last_n_epoch
        self.model_name = model_name
        self.mTrain_loss = math.inf
        self.mVal_metric = -math.inf
        self.cfg = cfg

    def after_train_epoch(self, runner):
        # TODO 发现一个奇怪问题，在本函数里面，_save函数调用的次数越多，速度越慢,特别是当模型很大时候
        self.save(runner, self.model_name, self.cfg)
        if self.mTrain_loss >= runner.epoch_train_loss:
            self.mTrain_loss = runner.epoch_train_loss
            runner.logger.debug('best train loss epoch ={}'.format(runner.epoch + 1))
            self.save(runner, 'train_best_model.pth', self.cfg)
        if self.every_n_epochs(runner, self.save_interval_epoch):
            self.save(runner, 'model_{}.pth'.format(runner.epoch + 1), self.cfg)
            del_path = os.path.join(runner.work_dir,
                                    'model_{}.pth'.format(
                                        runner.epoch + 1 - self.save_interval_epoch * self.last_n_epoch))
            if os.path.exists(del_path):
                os.remove(del_path)

    def after_val_epoch(self, runner):
        if runner.epoch_val_metric is not None and self.mVal_metric <= runner.epoch_val_metric:
            self.mVal_metric = runner.epoch_val_metric
            runner.logger.debug('best val metric epoch ={}'.format(runner.epoch))
            self.save(runner, 'train_val_model.pth', self.cfg)

    @staticmethod
    def save(runner, model_name, cfg=None):
        data = {"state_dict": runner.modeler.state_dict()}
        if runner.optimizer is not None:
            data["optimizer"] = runner.optimizer.state_dict()
        data["epoch"] = runner.epoch
        data['total_train_iter'] = runner.total_train_iter
        if runner.mode == 'train':
            data["loss"] = runner.epoch_train_loss
        else:
            data["metric"] = runner.epoch_val_metric

        if cfg is not None:
            data["cfg"] = cfg.text
        runner.logger.debug("Saving checkpoint to {}".format(os.path.join(runner.work_dir, model_name)))
        torch.save(data, os.path.join(runner.work_dir, model_name))
