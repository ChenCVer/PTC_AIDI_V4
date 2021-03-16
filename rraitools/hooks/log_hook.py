import torch
from .hook import Hook
from ..misc.timer import Timer, get_time_str
from .registry import HOOK


@HOOK.register_module()
class LogHook(Hook):
    def __init__(self, print_interval_iter=None):
        self._timer = Timer()
        self._print_interval_iter = print_interval_iter
        self._reset()

    def _reset(self):
        self._avg_val = None
        self._timer.reset()
        self._timer.tic()

    def before_train_epoch(self, runner):
        self._reset()
        runner.logger.debug('Starting epoch {}/{}.'.format(runner.epoch + 1, runner.max_epochs))

    def before_val_epoch(self, runner):
        self._reset()

    def after_train_iter(self, runner):
        if self._print_interval_iter is None:
            self._print_interval_iter = runner.print_interval_iter
        log_items = ['{}[{}/{}]'.format(runner.inner_iter + 1, runner.epoch + 1, runner.max_epochs)]
        log_str = ''
        for name, val in runner.batch_iter_output.items():
            if isinstance(val, list) or (isinstance(val, torch.Tensor) and val.ndim == 4):
                continue
            if isinstance(val, torch.Tensor):
                val = '{:.5f}'.format(val.item())
            else:
                val = '{:.5f}'.format(val)
            log_items.append('{}: {}'.format(name, val))
        log_str += ', '.join(log_items)
        if runner.inner_iter % self._print_interval_iter == 0:
            runner.logger.debug('{}'.format(log_str))

        _key_list = list(filter(lambda x: x.startswith('train_metric'), runner.batch_iter_output.keys()))
        if len(_key_list) == 0:
            self._avg_val = None
        else:
            if self._avg_val is None:
                self._avg_val = {key: 0. for key in _key_list}
            for k in self._avg_val.keys():
                self._avg_val[k] += runner.batch_iter_output[k]

    def after_val_iter(self, runner):
        if self._print_interval_iter is None:
            self._print_interval_iter = runner.print_interval_iter
        log_items = ['{}'.format(runner.inner_iter + 1)]
        log_str = ''
        for name, val in runner.batch_iter_output.items():
            if isinstance(val, list) or (isinstance(val, torch.Tensor) and val.ndim == 4):
                continue
            if isinstance(val, torch.Tensor):
                val = '{:.5f}'.format(val.item())
            else:
                val = '{:.5f}'.format(val)
            log_items.append('{}: {}'.format(name, val))
        log_str += ', '.join(log_items)
        if runner.inner_iter % self._print_interval_iter == 0:
            runner.logger.debug('{}'.format(log_str))

    def after_train_epoch(self, runner):
        self._timer.toc()
        if self._avg_val is not None:
            for key, value in self._avg_val.items():
                self._avg_val[key] = round(value / (runner.inner_iter+1), 5)
        eta = (runner.max_epochs - runner.epoch - 1) * self._timer.average_time
        eta_str = get_time_str(eta)
        runner.logger.info('Each epoch time={}s,eta={}'.format(round(self._timer.average_time, 5), eta_str))
        runner.logger.info(
            'Epoch finished ! mean all loss: {}'.format(round(runner.epoch_train_loss / (runner.inner_iter+1), 5)))
        if self._avg_val is not None:
            log_items = []
            log_str = ''
            for name, val in self._avg_val.items():
                log_items.append('{}: {}'.format(name, val))
            log_str += ', '.join(log_items)
            runner.logger.info('Epoch finished ! mean all metric: {}'.format(log_str))

    def after_val_epoch(self, runner):
        self._timer.toc()
        runner.logger.info('Each val epoch time={}s'.format(round(self._timer.average_time, 5)))
        if runner.epoch_val_loss != 0:
            runner.logger.info('Epoch finished ! Mean all val loss: {}'.format(round(runner.epoch_val_loss, 5)))
        runner.logger.info(
            'Epoch finished ! val metric: {}'.format(round(runner.epoch_val_metric / (runner.inner_iter+1), 5)))
