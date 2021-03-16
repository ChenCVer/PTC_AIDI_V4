from torch.nn.parallel import DataParallel, DistributedDataParallel

from mmcv.utils import Registry

MODULE_WRAPPERS = Registry('module metrics')
MODULE_WRAPPERS.register_module(module=DataParallel)
MODULE_WRAPPERS.register_module(module=DistributedDataParallel)
