# -*- coding:utf-8 -*-
from core.models import build_model
from mmcv import Config
from rraitools import Logger, EnvCollectHelper
from core.datasets import build_dataloader, build_dataset
from core.apis.inference import LoadImage
from core.datasets.pipelines import Compose
from core.apis import inference_detector, init_detector, show_result


if __name__ == '__main__':
    import torch

    # config = '../../../configs/cls.py'
    # config = '../../../configs/cls_cat_and_dog.py'
    config = '../../../configs/segmentation/seg_offline_magnetic_single.py'
    # config = '../../../configs/objectdetection.py'
    # config = '../../../configs/keypointdetection.py'
    cfg = Config.fromfile(config)
    Logger.init()
    Logger.info(EnvCollectHelper.collect_env_info())

    # Logger.debug(cfg)
    # Logger.info(cfg)
    # Logger.warn(cfg)
    # Logger.error(cfg)
    # Logger.critical(cfg)

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset,
                                   samples_per_gpu=cfg.data.eval_samples_per_gpu,
                                   workers_per_gpu=cfg.data.eval_workers_per_gpu,
                                   shuffle=False)

    # build the model and load checkpoint
    model = build_model(cfg.model, train_cfg=cfg.train_cfg,
                        valid_cfg=cfg.valid_cfg, test_cfg=cfg.test_cfg)

    print(model)
