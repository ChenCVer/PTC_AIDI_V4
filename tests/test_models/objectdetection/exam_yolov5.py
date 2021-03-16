# -*- coding:utf-8 -*-
from core.models import build_model
from mmcv import Config
from rraitools import Logger, EnvCollectHelper
from core.datasets import build_dataloader, build_dataset

if __name__ == '__main__':
    import torch
    config = '../../../configs/objectdetection/yolov5_csp_416_voc.py'
    cfg = Config.fromfile(config)
    Logger.init()
    Logger.info(EnvCollectHelper.collect_env_info())
    Logger.info(cfg)

    model = build_model(cfg.model)
    Logger.info(model)
    # 怎样可视化model
    model.eval()
    dataset = build_dataset(cfg.data.val)
    train_dataloader = build_dataloader(dataset,
                                        samples_per_gpu=cfg.data.eval_samples_per_gpu,
                                        workers_per_gpu=cfg.data.eval_workers_per_gpu,
                                        num_gpus=1,
                                        seed=0)

    for i, data_batch in enumerate(train_dataloader):
        out = model(data_batch, return_loss=False)

        if isinstance(out, list):
            for o in out:
                Logger.info(o.shape)
        else:
            Logger.info(out.shape)