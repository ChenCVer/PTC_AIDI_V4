# -*- coding:utf-8 -*-
import torch
from core.models import build_model
from mmcv import Config
from rraitools import Logger, EnvCollectHelper


if __name__ == '__main__':
    # config = '../../../configs/cls.py'
    config = '../../../configs/keypoints/centernet.py'
    # config = '../../../configs/classification/cls_cat_and_dog.py'
    # config = '../configs/sgd_linear.py'

    cfg = Config.fromfile(config)
    Logger.init()
    Logger.info(EnvCollectHelper.collect_env_info())

    model = build_model(cfg.model)
    model.eval()
    input_tensor = torch.rand((1, 3, 512, 512))
    out = model(input_tensor, return_loss=False)
    if isinstance(out, (list, tuple)):
        for o in out:
            Logger.info(o.shape)
    else:
        Logger.info(out.shape)