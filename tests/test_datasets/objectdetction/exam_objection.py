# -*- coding:utf-8 -*-
from mmcv import Config
from core.datasets import build_dataloader, build_dataset

if __name__ == '__main__':
    config = '../../../configs/objectdetection/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
    cfg = Config.fromfile(config)
    print(cfg)
    dataset = build_dataset(cfg.data.train)
    train_dataloader = build_dataloader(dataset,
                                        samples_per_gpu=cfg.data.samples_per_gpu,
                                        workers_per_gpu=cfg.data.workers_per_gpu,
                                        num_gpus=1,
                                        seed=0)

    for i, data_batch in enumerate(train_dataloader):
        print("data_batch = ", data_batch)