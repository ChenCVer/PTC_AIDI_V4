# -*- coding:utf-8 -*-
import numpy as np
from ...registry import PIPELINES
from rraitools import generate_heatmap


__all__ = ['GenerateHeatMap',]


@PIPELINES.register_module
class GenerateHeatMap(object):
    def __init__(self, sigma_xy, down_ratios, num_classes=None,
                 multi_scale=1.0, use_opencv=False, truncate_thre=0.0003,
                 is_all_in_one=False):
        if not isinstance(sigma_xy, (list, tuple)):
            sigma_xy = [sigma_xy]
        if not isinstance(down_ratios, (list, tuple)):
            down_ratios = [down_ratios]
        assert len(sigma_xy) == len(down_ratios)
        self.kernel_sizes = sigma_xy
        self.down_ratios = down_ratios
        self.num_classes = num_classes
        self.use_opencv = use_opencv  # 用opencv的高斯模糊进行操作.
        self.truncate_thre = truncate_thre
        self.multi_scale = multi_scale
        self.is_all_in_one = is_all_in_one  # 启用计数功能时, 所有关键点不再分类别.
        if self.is_all_in_one or num_classes is None:
            self.num_classes = 1

    def __call__(self, results):
        img = results['img']
        img_h, img_w = img.shape[:2]
        keypoint_label = results['keypoint_label']
        keypoint_label_c = keypoint_label.copy()
        keypoint_index = results.get('keypoint_index', None)
        heatmaps = []
        # TODO: 暂时不考虑多尺度.
        for sigma, down_ratio in zip(self.kernel_sizes, self.down_ratios):
            output_h = img_h // down_ratio
            output_w = img_w // down_ratio
            keypoint_label = keypoint_label_c / down_ratio
            heatmap = np.zeros((output_h, output_w, self.num_classes))
            for i, pt in enumerate(keypoint_label):
                output = generate_heatmap((output_h, output_w), pt, sigma,
                                          self.use_opencv, self.truncate_thre)
                if self.is_all_in_one:
                    output = np.where(heatmap[..., 0] > output, heatmap[..., 0], output)
                    heatmap[..., 0] = output
                else:
                    if keypoint_index is not None:
                        index = int(keypoint_index[i][0])
                        heatmap[..., index] = output
                    else:
                        heatmap[..., i] = output

            heatmaps.append(heatmap * self.multi_scale)

        # todo: 暂时不考虑多尺度, 所以这里取heatmaps[0]
        results['gt_heatmaps'] = heatmaps[0]

        return results