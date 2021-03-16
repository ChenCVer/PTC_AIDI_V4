# -*- coding:utf-8 -*-
import os
import numpy as np
from ...registry import PIPELINES


__all__ = [
    'LoadKeyPointFromFile',
]


@PIPELINES.register_module
class LoadKeyPointFromFile(object):

    def __init__(self):
        pass

    def __call__(self, results):
        label_path = results['label_path']
        key_points_list = []
        if not os.path.exists(label_path):
            assert FileNotFoundError, "File:{0} is not exist!".format(label_path)

        with open(label_path, "r") as file:
            for idx, line in enumerate(file):
                if idx == 0:
                    num_points = int(line.strip())
                else:
                    x1, y1 = [(t.strip()) for t in line.split()]
                    key_points_list.extend([float(x1), float(y1)])

        key_points_list = np.array(key_points_list, dtype=np.int).reshape(-1, 2)
        results["nums_keypoints"] = num_points
        results['keypoint_label'] = key_points_list

        return results