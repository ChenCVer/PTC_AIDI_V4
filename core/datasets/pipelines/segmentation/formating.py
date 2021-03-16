# -*- coding:utf-8 -*-
import numpy as np
from rraitools import encode_onehot
from ...registry import PIPELINES

try:
    from albumentations import Compose, BboxParams
except ImportError:
    Compose = None

__all__ = [
    'EncodeMaskToOneHot',
]


@PIPELINES.register_module
class EncodeMaskToOneHot(object):
    """
    此函数可以对mask进行编码. 该函数可以定制
    """
    def __init__(self, num_class=1, is_sigmoid=True, color_values=None):
        self.num_class = num_class
        self.color_values = color_values
        self.is_sigmoid = is_sigmoid

    def __call__(self, results):
        mask = results['gt_semantic_seg']

        # one-hot encode for every pixel
        if self.num_class > 1:
            mask = encode_onehot(mask, self.color_values)
            # mask = custom_encode_onehot(mask, self.color_values)
        else:
            if len(mask.shape) == 3:
                mask = mask[..., 0]
            if self.is_sigmoid:
                mask = mask / 255.0
            else:
                mask = np.where(mask > 0, 1.0, 0.0)[..., None]

        results['gt_semantic_seg'] = mask.squeeze()

        return results