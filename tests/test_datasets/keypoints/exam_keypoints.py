# -*- coding:utf-8 -*-
import cv2
import numpy as np
from mmcv import Config
from core.datasets import build_dataloader, build_dataset


if __name__ == '__main__':
    config = '../../../configs/_base_/datasets/keypoints/electronic_counter.py'
    cfg = Config.fromfile(config)
    print(cfg)
    dataset = build_dataset(cfg.data.train)
    train_dataloader = build_dataloader(dataset,
                                        samples_per_gpu=cfg.data.samples_per_gpu,
                                        workers_per_gpu=cfg.data.workers_per_gpu,
                                        num_gpus=1,
                                        seed=0)

    cv2.namedWindow("img", 0)
    cv2.namedWindow("label", 0)

    for i, data_batch in enumerate(train_dataloader):
        imgs = data_batch["img"].data[0].permute(0, 2, 3, 1).cpu().numpy()
        labels = data_batch["gt_heatmaps"].data[0].permute(0, 2, 3, 1).cpu().numpy()
        for idx in range(imgs.shape[0]):
            img = np.uint8(imgs[idx, ...] * 255)
            label = np.uint8(labels[idx, :, :, 0]*255)
            print("img.shape = ", img.shape)
            print("label.shape = ", label.shape)
            cv2.imshow("img", img)
            cv2.imshow("label", label)
            cv2.waitKey(0)
        print("--------------------")