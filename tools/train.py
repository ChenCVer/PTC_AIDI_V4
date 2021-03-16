
import os
import copy
import ssl
import time
import mmcv
import torch
import argparse
import os.path as osp
from mmcv import Config, DictAction

from core import __version__
from rraitools import EnvCollectHelper
from core.models import build_model
from core.datasets import build_dataset
from core.apis import set_random_seed, train_model
from core.utils import collect_env, get_root_logger

ssl._create_default_https_context = ssl._create_unverified_context


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--resume-from', help='the checkpoint file to resume from')
    parser.add_argument('--no-evaluate', action='store_true',
                        help='whether not to evaluate the checkpoint during training')
    parser.add_argument('--use_gpu', type=bool, default=True, help='is use gpu')

    group_gpus = parser.add_mutually_exclusive_group()

    group_gpus.add_argument('--gpus',
                            type=int,
                            help='number of gpus to use (only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use (only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')  # 固定随机数.
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')

    parser.add_argument('--options', nargs='+', action=DictAction, help='arguments in dict')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')

    parser.add_argument('--local_rank', type=int, default=0)

    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    # 1.配置文件处理
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)

    # 2.gpu相关设置
    if args.use_gpu:
        EnvCollectHelper.set_environ(args.gpus)

    # 3.工作路径设置(动态以配置文件生成一个专用文件夹)
    cfg.work_dir = osp.join(cfg.work_dir, osp.splitext(osp.basename(args.config))[0])
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))

    # 4. cudnn_benchmark设置
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # 5. resume 和 gpu_ids设定
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    cfg.log_config['log_file'] = log_file
    logger = get_root_logger(**cfg.log_config)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 80 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed

    # build model
    model = build_model(cfg.model, train_cfg=cfg.train_cfg,
                        valid_cfg=cfg.valid_cfg, test_cfg=cfg.test_cfg)

    # TODO: build dataset, 先建立train, 然后在建立val数据集, 官方默认这样, 需要加assert强行保证
    assert "train" in cfg.workflow[0], "train mode must be first in workflow!"
    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        datasets.append(build_dataset(val_dataset))

    if cfg.checkpoint_config is not None:
        # save core version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            core_version=__version__,
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES)

    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    train_model(
        model,
        datasets,
        cfg,
        evaluate=(not args.no_evaluate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
