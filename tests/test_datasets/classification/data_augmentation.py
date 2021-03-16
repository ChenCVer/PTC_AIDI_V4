# -*- coding:utf-8 -*-
import cv2
import numpy as np
from mmcv import Config
from core.datasets import build_dataloader, build_dataset

if __name__ == '__main__':
    config = '../../../configs/classification/cls_ice-cream-bar.py'
    cfg = Config.fromfile(config)
    dataset = build_dataset(cfg.data.train)
    train_dataloader = build_dataloader(dataset,
                                        samples_per_gpu=cfg.data.samples_per_gpu,
                                        workers_per_gpu=cfg.data.workers_per_gpu,
                                        num_gpus=1,
                                        seed=0)

    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    for i, data_batch in enumerate(train_dataloader):
        img_data = data_batch["img"].data[0]
        for idx, data in enumerate(img_data):
            img = img_data[idx, ...].numpy().transpose(1, 2, 0)
            img = np.uint8(img * 255.0)
            print("img.shape = ", img.shape)
            cv2.imshow("img", img)
            cv2.waitKey(0)
