# Copyright (c) Open-MMLab. All rights reserved.
import os.path as osp

from mmcv.utils import TORCH_VERSION
from ...dist_utils import master_only
from ..hook import HOOKS
from .base import LoggerHook


@HOOKS.register_module()
class TensorboardLoggerHook(LoggerHook):

    def __init__(self,
                 log_dir=None,
                 interval=10,
                 ignore_last=True,
                 reset_flag=True,
                 by_epoch=True):
        super(TensorboardLoggerHook, self).__init__(interval, ignore_last,
                                                    reset_flag, by_epoch)
        self.log_dir = log_dir

    @master_only
    def before_run(self, runner):
        if TORCH_VERSION < '1.1' or TORCH_VERSION == 'parrots':
            try:
                from tensorboardX import SummaryWriter
            except ImportError:
                raise ImportError('Please install tensorboardX to use '
                                  'TensorboardLoggerHook.')
        else:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError:
                raise ImportError(
                    'Please run "pip install future tensorboard" to install '
                    'the dependencies to use torch.utils.tensorboard '
                    '(applicable to PyTorch 1.1 or higher)')

        if self.log_dir is None:
            self.log_dir = osp.join(runner.work_dir, 'tf_logs')
        self.writer = SummaryWriter(self.log_dir)

    @master_only
    def log(self, runner):
        for var in runner.log_buffer.output:
            if var in ['time', 'data_time']:
                continue
            tag = f'{var}/{runner.mode}'
            record = runner.log_buffer.output[var]
            if isinstance(record, str):
                self.writer.add_text(tag, record, runner.iter)
            else:
                self.writer.add_scalar(tag, runner.log_buffer.output[var],
                                       runner.iter)
        # add learning rate
        lrs = runner.current_lr()
        if isinstance(lrs, dict):
            for name, value in lrs.items():
                self.writer.add_scalar(f'learning_rate/{name}', value[0],
                                       runner.iter)
        else:
            self.writer.add_scalar('learning_rate', lrs[0], runner.iter)
        # add momentum
        momentums = runner.current_momentum()
        if isinstance(momentums, dict):
            for name, value in momentums.items():
                self.writer.add_scalar(f'momentum/{name}', value[0],
                                       runner.iter)
        else:
            self.writer.add_scalar('momentum', momentums[0], runner.iter)

    def after_run(self, runner):
        self.writer.close()

    def after_train_iter(self, runner):
        """
        Notes: 添加相关imgs信息到tensorboard中
        """
        if self.by_epoch and self.every_n_inner_iters(runner, self.interval):
            runner.log_buffer.average(self.interval)
        elif not self.by_epoch and self.every_n_iters(runner, self.interval):
            runner.log_buffer.average(self.interval)
        elif self.end_of_epoch(runner) and not self.ignore_last:
            # not precise but more stable
            runner.log_buffer.average(self.interval)

        if runner.log_buffer.ready:
            self.log(runner)
            if self.reset_flag:
                runner.log_buffer.clear_output()
