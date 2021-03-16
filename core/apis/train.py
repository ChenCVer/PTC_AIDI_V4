import copy
import random

import numpy as np
import torch
from mmcv.parallel import MMDataParallel
from mmcv.runner import (EpochBasedRunner, OptimizerHook, build_optimizer)
from core.core import EvalHook, Fp16OptimizerHook
from core.datasets import build_dataloader, build_dataset
from core.utils import get_root_logger


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_model(model,
                dataset,
                cfg,
                evaluate=False,
                timestamp=None,
                meta=None):
    logger = get_root_logger(cfg.log_config.log_level)
    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    if 'imgs_per_gpu' in cfg.data:
        logger.warning('"imgs_per_gpu" is deprecated in MMDet V2.0. '
                       'Please use "samples_per_gpu" instead')
        if 'samples_per_gpu' in cfg.data:
            logger.warning(
                f'Got "imgs_per_gpu"={cfg.data.imgs_per_gpu} and '
                f'"samples_per_gpu"={cfg.data.samples_per_gpu}, "imgs_per_gpu"'
                f'={cfg.data.imgs_per_gpu} is used in this experiments')
        else:
            logger.warning(
                'Automatically set "samples_per_gpu"="imgs_per_gpu"='
                f'{cfg.data.imgs_per_gpu} in this experiments')
        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu

    # 此处需要修改, 如果分割是在线训练, 比如小图训练(train), 整图测试(val和test),
    # 则samples_per_gpu是肯定不一样的, train和val的batch_size不一样
    data_loaders = []
    train_dataloader = build_dataloader(dataset[0],
                                        samples_per_gpu=cfg.data.samples_per_gpu,
                                        workers_per_gpu=cfg.data.workers_per_gpu,
                                        num_gpus=len(cfg.gpu_ids),
                                        seed=cfg.seed)

    data_loaders.append(train_dataloader)

    # TODO: 走val_step()路线, 需要后期实现自适应策略, 考虑小图训练, 整图测试
    if len(dataset) == 2:
        assert cfg.data.eval_samples_per_gpu == 1, "cfg.data.eval_samples_per_gpu must be equal 1."
        val_dataloader = build_dataloader(dataset[1],
                                          samples_per_gpu=cfg.data.eval_samples_per_gpu,
                                          workers_per_gpu=cfg.data.eval_workers_per_gpu,
                                          num_gpus=len(cfg.gpu_ids),
                                          shuffle=False,
                                          seed=cfg.seed)
        data_loaders.append(val_dataloader)

    model = MMDataParallel(model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)
    # build optimizer and runner
    optimizer = build_optimizer(model, cfg.optimizer)
    # init runner
    runner = EpochBasedRunner(
        cfg,
        model,
        optimizer=optimizer,
        work_dir=cfg.work_dir,
        logger=logger,
        meta=meta)
    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    optimizer_config = cfg.optimizer_config
    # register hooks, 注册各种钩子, 以便获取中间变量
    runner.register_training_hooks(cfg.lr_config,
                                   optimizer_config,
                                   cfg.checkpoint_config,
                                   cfg.log_config,
                                   cfg.get('momentum_config', None))

    # register eval hooks, val_dataset用作evaluationHook用
    # 在验证集上给出相应的评估指标(mAP, mIOU, acc等), 以便指导用户参考, forward_test()路线
    if evaluate:
        eval_dataset = copy.deepcopy(cfg.data.val)
        eval_dataset.pipeline = cfg.data.test.pipeline
        # test_mode=True, 不加载label/gt.
        eval_dataset = build_dataset(eval_dataset, dict(test_mode=True))
        eval_dataloader = build_dataloader(eval_dataset,
                                           samples_per_gpu=cfg.data.eval_samples_per_gpu,
                                           workers_per_gpu=cfg.data.eval_workers_per_gpu,
                                           shuffle=False)
        eval_cfg = cfg.get('evaluation', {})
        runner.register_hook(EvalHook(eval_dataloader, **eval_cfg))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)

    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)