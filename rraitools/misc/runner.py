import time
import warnings
import torch
from .base_runner import BaseRunner
from .misc import is_list_of
from .logger import Logger
from . import checkpoint
from .checkpoint import _load_checkpoint_file
from ..hooks import CheckpointerHook


def assert_dict(clazz):
    if not isinstance(clazz, dict):
        raise TypeError('clazz() must return a dict')


# 暂时不用
def get_max_memory_allocated():
    mem = torch.cuda.max_memory_allocated()
    mem_mb = torch.tensor([mem / (1024 * 1024)],
                          dtype=torch.int,
                          device=torch.device('cuda'))
    return mem_mb.item()


class Runner(BaseRunner):
    """A training helper for PyTorch.

    Args:
        modeler (:obj:`torch.nn.Module`): The model to be run.
        optimizer (dict or :obj:`torch.optim.Optimizer`): If it is a dict,
            runner will construct an optimizer according to it.
        logger (:obj:`logging.Logger`): Custom logger. If `None`, use the
            default logger.
        hooks (list)
        meta (dict | None): A dict records some import information such as
            environment info and seed, which will be logged in logger hook.
    """

    def __init__(self,
                 modeler,
                 optimizer,
                 hooks,
                 work_dir,
                 print_interval_iter,
                 logger=Logger,
                 meta=None):
        super(Runner, self).__init__(modeler,
                                     optimizer,
                                     hooks=hooks,
                                     print_interval_iter=print_interval_iter,
                                     work_dir=work_dir,
                                     logger=logger,
                                     meta=meta)
        self._stop = False

    def resume(self,
               filename,
               resume_optimizer=True,
               map_location='default'):
        if map_location == 'default':
            device_id = torch.cuda.current_device()
            checkpoint = _load_checkpoint_file(filename,
                                               map_location=lambda storage, loc: storage.cuda(device_id))
        else:
            checkpoint = _load_checkpoint_file(filename, map_location=map_location)
        self.load_checkpoint(filename, strict=True)
        self._epoch = checkpoint['epoch']
        self._total_train_iter = checkpoint['total_train_iter']
        if 'optimizer' in checkpoint and resume_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info('resumed epoch {}, total_train_iter {}'.format(self._epoch, self._total_train_iter))

    def load_checkpoint(self, filename, map_location='cpu', strict=False):
        self.logger.info('load checkpoint from {}'.format(filename))
        return checkpoint.load_checkpoint(self.modeler, filename, map_location, strict, self.logger)

    def update_kwargs(self, **kwargs):
        # todo update kwargs by property
        for attr_name in dir(Runner):
            attr = getattr(Runner, attr_name)
            if isinstance(attr, property):
                if attr in kwargs:
                    warnings.warn('kwargs key confict, please check key: {}'.format(attr))
                kwargs[attr_name] = self.__getattribute__(attr_name)
        return kwargs

    def train(self, data_loader, batch_processor, **kwargs):
        self.modeler.train()  # 切换网络至train模式
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(data_loader)
        self.call_hook('before_train_epoch')
        self._epoch_train_loss = 0
        assert len(data_loader) > 0, 'dataset length is less than batch size'
        for i, data_batch in enumerate(data_loader):
            # 早停策略
            if self._stop is True:
                self._save_checkpoint_stoptime()
                break
            self._data_batch = data_batch
            current_count = data_batch[0].shape[0]
            self._inner_iter = i
            self.call_hook('before_train_iter')
            kwargs = self.update_kwargs(**kwargs)  # 数据更新操作
            batch_iter_output = batch_processor.execute(self.modeler, **kwargs)
            if not isinstance(batch_iter_output, dict):
                raise TypeError('batch_processor() must return a dict')
            assert 'batch_loss' in batch_iter_output.keys()
            self._batch_iter_output = batch_iter_output
            self._epoch_train_loss += (batch_iter_output['batch_loss'].item() / current_count)
            self._total_train_iter += 1
            self.call_hook('after_train_iter')
        self.call_hook('after_train_epoch')
        self._epoch += 1

    def val(self, data_loader, batch_processor, **kwargs):
        Logger.debug('-----start into val model(sample len={},batch len={})------'.format(len(data_loader.dataset),
                                                                                          len(data_loader)))
        self.modeler.eval()
        self.mode = 'val'
        self.data_loader = data_loader
        self._epoch_val_metric = 0
        self._epoch_val_loss = 0
        self.call_hook('before_val_epoch')
        assert len(data_loader) > 0, 'dataset length is less than batch size'
        for i, data_batch in enumerate(data_loader):
            if self._stop is True:
                self._save_checkpoint_stoptime()
                break
            self._data_batch = data_batch
            current_count = data_batch[0].shape[0]
            self._inner_iter = i
            self.call_hook('before_val_iter')
            kwargs = self.update_kwargs(**kwargs)
            with torch.no_grad():
                batch_iter_output = batch_processor.execute(self.modeler, **kwargs)
            if not isinstance(batch_iter_output, dict):
                raise TypeError('batch_processor() must return a dict')
            self._epoch_val_loss += (batch_iter_output['batch_loss'].item() / current_count)
            assert 'batch_metric' in batch_iter_output.keys()
            batch_metric = batch_iter_output['batch_metric']
            if isinstance(batch_metric, torch.Tensor):
                batch_metric = batch_metric.item()
            if 'batch_loss' in batch_iter_output.keys():
                self._epoch_val_loss += (batch_iter_output['batch_loss'].item() / current_count)
            self._epoch_val_metric += batch_metric
            self._batch_iter_output = batch_iter_output
            self._total_val_iter += 1
            self.call_hook('after_val_iter')
        self.call_hook('after_val_epoch')

    def run(self,
            data_loaders,
            batch_processors,
            workflow,
            max_epochs, **kwargs):
        """
        Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.
            max_epochs (int): Total training epochs.
        """
        assert isinstance(data_loaders, list)
        assert isinstance(batch_processors, list)
        assert is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)
        assert len(batch_processors) == len(workflow)

        self._max_epochs = max_epochs
        self.call_hook('before_run')

        while self.epoch < max_epochs:
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                if isinstance(mode, str):  # self.train()
                    if not hasattr(self, mode):
                        raise ValueError(
                            'runner has no method named "{}" to run an epoch'.format(mode))
                    epoch_runner = getattr(self, mode)  # mode字符转为函数变量
                elif callable(mode):  # custom train()
                    epoch_runner = mode
                else:
                    raise TypeError('mode in workflow must be a str or '
                                    'callable function, not {}'.format(type(mode)))
                for _ in range(epochs):
                    if mode == 'train' and self.epoch >= max_epochs:
                        return
                    epoch_runner(data_loaders[i], batch_processors[i], **kwargs)
                    if self._stop is True:
                        # self._save_checkpoint_stoptime()
                        break
                if self._stop is True:
                    # self._save_checkpoint_stoptime()
                    break
            if self._stop is True:
                break

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')

    def stop(self):
        self._stop = True

    def _save_checkpoint_stoptime(self):
        CheckpointerHook.save(self, 'model_interrupt.pth')
