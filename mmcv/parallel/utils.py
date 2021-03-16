# Copyright (c) Open-MMLab. All rights reserved.
from .registry import MODULE_WRAPPERS


def is_module_wrapper(module):
    """Check if a module is a module metrics.

    The following 3 modules in MMCV (and their subclasses) are regarded as
    module wrappers: DataParallel, DistributedDataParallel,
    MMDistributedDataParallel (the deprecated version). You may add you own
    module metrics by registering it to mmcv.parallel.MODULE_WRAPPERS.

    Args:
        module (nn.Module): The module to be checked.

    Returns:
        bool: True if the input module is a module metrics.
    """
    module_wrappers = tuple(MODULE_WRAPPERS.module_dict.values())
    return isinstance(module, module_wrappers)
