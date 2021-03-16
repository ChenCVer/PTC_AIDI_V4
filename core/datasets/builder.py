# -*- coding:utf-8 -*-
import copy
import random
import platform
import resource
import numpy as np
import rraitools
from functools import partial
from .registry import DATASETS
from rraitools.misc import collate
from mmcv.runner import get_dist_info
from core.utils import build_from_cfg
from torch.utils.data import DataLoader
from core.datasets.data_loader import (ConcatDataset, RepeatDataset,
                                       ClassBalancedDataset)
from .samplers import GroupSampler

if platform.system() != 'Windows':
    # https://github.com/pytorch/pytorch/issues/973
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    hard_limit = rlimit[1]
    soft_limit = min(4096, hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))


# 参考网页: https://www.jb51.net/article/180952.htm
# DataLoader中的参数: 用户定义的每个worker初始化的时候需要执行的函数。
# 如果dataloader采用了多线程(num_workers > 1), 那么由于读取数据的顺序不同, 最终运行结果也会有差异.
# 也就是说, 改变num_workers参数,也会对实验结果产生影响. 目前暂时没有发现解决这个问题的方法, 但是只要
# 固定num_workers数目(线程数)不变, 基本上也能够重复实验结果.
def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _concat_dataset(cfg, default_args=None):
    ann_files = cfg['ann_file']
    img_prefixes = cfg.get('img_prefix', None)
    seg_prefixes = cfg.get('seg_prefix', None)
    proposal_files = cfg.get('proposal_file', None)

    datasets = []
    num_dset = len(ann_files)

    for i in range(num_dset):
        data_cfg = copy.deepcopy(cfg)
        data_cfg['ann_file'] = ann_files[i]
        if isinstance(img_prefixes, (list, tuple)):
            data_cfg['img_prefix'] = img_prefixes[i]
        if isinstance(seg_prefixes, (list, tuple)):
            data_cfg['seg_prefix'] = seg_prefixes[i]
        if isinstance(proposal_files, (list, tuple)):
            data_cfg['proposal_file'] = proposal_files[i]

        datasets.append(build_dataset(data_cfg, default_args))

    return ConcatDataset(datasets)


def build_dataset(cfg, default_args=None):
    """
    Noets: 这里build_dataset包括复合类型的ConcatDataSet, RepeatData, ClassBalanceDataset
           以及不是复合类型的比如VOCDataSet, CocoDataSet等等.
    """
    # ConcatDataset
    if isinstance(cfg, (list, tuple)):
        dataset = ConcatDataset([build_dataset(c, default_args) for c in cfg])
    # ReapDataset
    elif cfg['type'] == 'RepeatDataset':
        dataset = RepeatDataset(
            build_dataset(cfg['dataset'], default_args), cfg['times'])
    # ClassBalanceDataset
    elif cfg['type'] == 'ClassBalancedDataset':
        dataset = ClassBalancedDataset(
            build_dataset(cfg['dataset'], default_args), cfg['oversample_thr'])
    elif isinstance(cfg.get('ann_file'), (list, tuple)):
        dataset = _concat_dataset(cfg, default_args)
    else:
        dataset = build_from_cfg(cfg, DATASETS, default_args)

    return dataset


def build_dataloader(dataset,
                     samples_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     shuffle=True,
                     seed=None,
                     **kwargs):
    """Build PyTorch DataLoader.

    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader for all GPUs.

    Args:
        dataset (Dataset): A PyTorch dataset.
        samples_per_gpu (int): Number of training samples on each GPU, i.e.,
            batch size of each GPU.
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU.
        num_gpus (int): Number of GPUs. Only used in non-distributed training.
        dist (bool): Distributed training/test or not. Default: True.
        shuffle (bool): Whether to shuffle the data at every epoch.
            Default: True.
        kwargs: any keyword argument to be used to initialize DataLoader

    Returns:
        DataLoader: A PyTorch dataloader.
    """
    rank, world_size = get_dist_info()
    # TODO: 这里就需要判断下, 由于分类和分割没有flag选项, 但同时也需要shuffle
    if hasattr(dataset, 'flag') and dataset.flag is not None:
        sampler = GroupSampler(dataset, samples_per_gpu) if shuffle else None
    else:
        sampler = None

    batch_size = num_gpus * samples_per_gpu
    num_workers = num_gpus * workers_per_gpu

    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None

    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             sampler=sampler,
                             num_workers=num_workers,
                             collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
                             pin_memory=False,
                             shuffle=False if sampler is not None else shuffle,  # TODO: 细节待验证.
                             worker_init_fn=init_fn,
                             **kwargs)

    return data_loader
