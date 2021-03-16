# -*- coding:utf-8 -*-
import numpy as np
import os
from rraitools import ImageHelper
from ...registry import PIPELINES

try:
    from albumentations import Compose, BboxParams
except ImportError:
    Compose = None


__all__ = [
    'LoadMaskFromFile',
]


@PIPELINES.register_module
class LoadMaskFromFile(object):
    # todo: 2021-01-14: 后续考虑与LoadAnnotations是否可以合并.
    def __init__(self, num_class=1, is_sigmoid=True):
        self.num_class = num_class
        self.is_sigmoid = is_sigmoid

    def __call__(self, results):
        path = results['path']
        gt_labels = results['label']
        root, fname = os.path.splitext(path)
        mask_path = results['label_path']
        # gt_labels == 0 means mask is all zero.
        if not os.path.exists(mask_path) and gt_labels == 0:
            # 可以不需要label
            image = results['img']
            if isinstance(image, list):
                h, w = image[0].shape[:2]
            else:
                h, w = image.shape[:2]
            mask = np.zeros((h, w, 3), np.uint8)
        else:
            if self.num_class == 1:
                if fname == '.npy':
                    mask = np.load(mask_path).astype(np.uint8)
                    mask = ImageHelper.convert_bgr_to_gray(mask)
                else:
                    mask = ImageHelper.read_img(mask_path, 'GRAY')

                if self.is_sigmoid:
                    mask = mask
                else:
                    mask[mask > 0] = 255
                mask = np.tile(mask, [3, 1, 1]).transpose([1, 2, 0])
            else:
                if fname == '.npy':
                    mask = np.load(mask_path).astype(np.uint8)
                else:
                    mask = ImageHelper.read_img(mask_path)
        results['gt_semantic_seg'] = mask
        results['mask_path'] = mask_path
        results['seg_fields'] = ['gt_semantic_seg']  # 后续数据增强需要用到key: seg_fields

        return results