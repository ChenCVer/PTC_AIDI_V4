from abc import abstractmethod, ABCMeta


class BaseRunner(object, metaclass=ABCMeta):
    """A training helper for PyTorch.

    Args:
        modeler (:obj:`torch.nn.Module`): The model to be run.
        optimizer (dict or :obj:`torch.optim.Optimizer`): If it is a dict,
            runner will construct an optimizer according to it.
        work_dir (str, optional): The working directory to save checkpoints
            and logs.
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
                 logger=None,
                 meta=None):
        assert isinstance(hooks, list), 'hooks must be a list'

        self.modeler = modeler
        self.optimizer = optimizer
        self.logger = logger
        self.work_dir = work_dir
        self.print_interval_iter = print_interval_iter

        # get modeler name from the model class
        if hasattr(self.modeler, 'module'):
            self._model_name = self.modeler.module.__class__.__name__
        else:
            self._model_name = self.modeler.__class__.__name__

        if meta is not None:
            assert isinstance(meta, dict), '"meta" must be a dict or None'
        self.meta = meta

        self.mode = None
        self._hooks = hooks
        self._data_batch = None
        self._epoch = 0
        self._total_train_iter = 0
        self._total_val_iter = 0
        self._inner_iter = 0
        self._max_epochs = 0
        self._max_iters = 0
        self._epoch_train_loss = 0
        self._epoch_train_metric = None
        self._epoch_val_loss = 0
        self._epoch_val_metric = None
        self._batch_iter_output = {}

    @property
    def data_batch(self):
        return self._data_batch

    @property
    def epoch_val_loss(self):
        return self._epoch_val_loss

    @property
    def epoch_val_metric(self):
        return self._epoch_val_metric

    @property
    def epoch_train_loss(self):
        return self._epoch_train_loss

    @property
    def epoch_train_metric(self):
        return self._epoch_train_metric

    @property
    def model_name(self):
        """str: Name of the model, usually the module class name."""
        return self._model_name

    @property
    def hooks(self):
        """list[:obj:`Hook`]: A list of registered hooks."""
        return self._hooks

    @property
    def epoch(self):
        """int: Current epoch."""
        return self._epoch

    @property
    def total_train_iter(self):
        """int: Current iteration."""
        return self._total_train_iter

    @property
    def total_val_iter(self):
        """int: Current iteration."""
        return self._total_val_iter

    @property
    def inner_iter(self):
        """int: Iteration in an epoch."""
        return self._inner_iter

    @property
    def batch_iter_output(self):
        return self._batch_iter_output

    @property
    def max_epochs(self):
        """int: Maximum training epochs."""
        return self._max_epochs

    @property
    def max_iters(self):
        """int: Maximum training iterations."""
        return self._max_iters

    def call_hook(self, fn_name):
        """
        func: 此函数的功能是根据fn_name, 比如"before_run", 则
        :param fn_name: 函数名字符串
        :return:每hook类都有实现fn_name的方法, getattr(hook, fn_name)(self)
                就是在调用hook的fn_name方法,并将self作为实参.
        """
        for hook in self._hooks:
            getattr(hook, fn_name)(self)

    @abstractmethod
    def train(self, data_loader, batch_porcess, **kwargs):
        pass

    @abstractmethod
    def val(self, data_loader, batch_porcess, **kwargs):
        pass

    @abstractmethod
    def run(self, data_loaders, batch_porcesses, workflow, max_epochs, **kwargs):
        pass
