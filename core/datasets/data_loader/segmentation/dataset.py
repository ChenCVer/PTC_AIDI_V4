# -*- coding:utf-8 -*-
import cv2
import numpy as np
from core.datasets.registry import DATASETS
from ..base.base_dataset import BaseDataset
from ..utils import make_dataset, find_classes, ratio_make_dataset
from core.core import eval_seg_metrics


__all__ = ['SegDataset',
           ]

# 默认后缀, 后期需要优化, 尽量写在配置文件中。
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp',)
IMG_EXCLUDE_EXTENSIONS = ('_mask.jpg', '_mask.jpeg', '_mask.png', '_mask.bmp')


@DATASETS.register_module()
class SegDataset(BaseDataset):

    def __init__(self,
                 img_prefix,  # 数据存放路径根文件夹
                 pipeline,    # 数据增强pipeline
                 gather_flag=False,  # 是否控制batchsize内部比例.
                 label_endswith=None,
                 test_mode=False):

        assert img_prefix is not None, "img_prefix must be not None!"
        assert pipeline is not None, "pipeline must be not None!"

        self.gather_flag = gather_flag
        self.label_endswith = label_endswith
        self.classes, self.class_to_idx = find_classes(img_prefix)
        super(SegDataset, self).__init__(img_prefix, pipeline)

    def load_annotations(self, *args, **kwargs):
        """
        Notes: 核心函数.
        """
        # 获取所有的sample样本
        if self.gather_flag:
            samples = ratio_make_dataset(self.img_prefix,
                                         self.class_to_idx,
                                         label_endswith=self.label_endswith,
                                         extensions=self.extensions,
                                         exclude_extensions=self.exclude_extensions,
                                         is_valid_file=True)
        else:
            samples = make_dataset(self.img_prefix,
                                   self.class_to_idx,
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
        if self.gather_flag:
            self.datalen = 0
            return self._get_dataset_len(self.data_infos)  # self.data_infos此时为字典
        else:
            return len(self.data_infos)  # self.data_infos此时为列表

    def __getitem__(self, idx):
        if self.gather_flag:  # gather_flag=True, batch类控制比例.
            data_unit = self.data_infos.copy()
        else:
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
                 **kwargs):

        seg_allowed_metrics = ['soft-IOU', 'hard-IOU']

        if metric is None:
            raise ValueError("metric must be not NoneType")

        if not isinstance(metric, list):
            raise TypeError("metric must be list Type")

        for metc in metric:
            if metc not in seg_allowed_metrics:
                raise KeyError('metc: {} is not supported, '
                               '{} is can be supported'.format(metc, seg_allowed_metrics))

        assert len(metric) == 1, "metric must be only one, soft-IOU or hard-IOU"
        if 'soft-IOU' in metric:
            score_thr = cfg.get("test_cfg", None).get("score_thr", 0.0)
            assert score_thr is None, "if soft-IOU in metric, then score_thr must be None!"

        # TODO: 对于分割评估来说, label是图片, 要考虑GPU显存占用问题, 如果VAL_DATA数据流太大,
        #  需要设定上限(随机获取).

        targets_paths = [value["label_path"] for value in self.data_infos]
        targets = list(map(cv2.imread, targets_paths))
        return eval_seg_metrics(results, targets, metric, cfg=cfg)