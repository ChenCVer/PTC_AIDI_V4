# -*- coding:utf-8 -*-
import cv2
import numpy as np
from core.datasets.registry import DATASETS
from ..base.base_dataset import BaseDataset
from ..utils import make_dataset_for_keypoints
from core.core import eval_seg_metrics


__all__ = ['KeyPointDataset',
           ]


@DATASETS.register_module()
class KeyPointDataset(BaseDataset):

    def __init__(self,
                 img_prefix,  # 数据存放路径根文件夹
                 pipeline,    # 数据增强pipeline
                 label_endswith,
                 test_mode=False):

        assert img_prefix is not None, "img_prefix must be not None"
        assert pipeline is not None, "pipeline must be not None!"
        self.label_endswith = label_endswith
        super(KeyPointDataset, self).__init__(img_prefix, pipeline)

    def load_annotations(self, *args, **kwargs):
        """
        Notes: 核心函数.
        """
        # 获取所有的sample样本
        samples = make_dataset_for_keypoints(self.img_prefix,
                                             label_endswith=self.label_endswith,
                                             extensions=self.extensions,
                                             exclude_extensions=self.exclude_extensions,
                                             is_valid_file=True)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.data_prefix + "\n"
                                "Supported extensions are: " + ",".join(self.extensions)))

        return samples

    def _get_dataset_len(self, data_dict):
        """
        TODO:后期需要考虑_class_ratios_dict中的比例系数含有0的情况, 暂时只考虑计算全部数据集的长度
        """
        for key, value in data_dict.items():
            if isinstance(value, list):
                self.datalen += len(value)
            else:
                self._get_dataset_len(value)

        return self.datalen

    def get_cat_ids(self, idx):
        return self.data_infos[idx]['cls_id'].astype(np.int).tolist()

    def __len__(self):
        return len(self.data_infos)  # self.data_infos此时为列表

    def __getitem__(self, idx):
        # 获取数据
        data_unit = self.getitem(idx).copy()
        # 数据增强
        data_unit = self.pipeline(data_unit)

        return data_unit

    def getitem(self, idx):
        return self.data_infos[idx]

    def evaluate(self,
                 results,
                 metric,
                 cfg=None,
                 logger=None,
                 use_sigmoid=False,
                 thres_score=None,
                 **kwargs):

        if metric is None:
            raise ValueError("metric must be not NoneType")

        if not isinstance(metric, list):
            raise TypeError("metric must be list Type")

        seg_allowed_metrics = ['soft-IOU', 'hard-IOU', 'soft-pa', 'hard-pa']
        # TODO: 对于分割评估来说, label是图片, 要考虑GPU显存占用问题, 如果VAL_DATA数据流太大,
        #  需要设定上限(随机获取).
        for metc in metric:
            if metc not in seg_allowed_metrics:
                raise KeyError('metc: {} is not supported, '
                               '{} is can be supported'.format(metc, seg_allowed_metrics))
        targets_paths = [value["label_path"] for value in self.data_infos]
        targets = list(map(cv2.imread, targets_paths))
        return eval_seg_metrics(results, targets, metric, thres_score, cfg=cfg)