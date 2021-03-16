# -*- coding:utf-8 -*-
from core.datasets.registry import DATASETS
from ..base.base_dataset import BaseDataset
from ..utils import make_dataset, find_classes, ratio_make_dataset
from core.core import eval_cls_metrics

__all__ = ['ClsDataset',]


@DATASETS.register_module()
class ClsDataset(BaseDataset):

    def __init__(self,
                 img_prefix,  # 数据存放路径根文件夹
                 pipeline,  # 数据增强pipeline
                 gather_flag=False,  # 是否控制batchsize内部比例.
                 test_mode=False):

        self.gather_flag = gather_flag
        assert pipeline is not None, "pipe must be not None!"
        self.classes, self.class_to_idx = find_classes(img_prefix)
        super(ClsDataset, self).__init__(img_prefix, pipeline)

    def load_annotations(self, *args, **kwargs):
        """
        Notes: 核心函数.
        """
        # 获取所有的sample样本
        if self.gather_flag:  # 控制比例
            samples = ratio_make_dataset(self.img_prefix,
                                         self.class_to_idx,
                                         extensions=self.extensions,
                                         exclude_extensions=self.exclude_extensions,
                                         is_valid_file=True)
        else:
            samples = make_dataset(self.img_prefix,
                                   self.class_to_idx,
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
        """
        获取dataset中所有类的类别索引.
        """
        return list([self.data_infos[idx]['cls_id']])

    def __len__(self):
        if self.gather_flag:
            self.datalen = 0
            return self._get_dataset_len(self.data_infos)  # self.data_infos此时为字典
        else:
            return len(self.data_infos)  # self.data_infos此时为列表

    def __getitem__(self, idx):
        if self.gather_flag:
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
                 use_sigmoid=False,
                 thres_score=None,
                 **kwargs):

        cls_allowed_metrics = ['acc', 'precision', 'recall',
                               'f1-score', 'confusion_matrix']

        if metric is None:
            raise ValueError("metric must be not NoneType")

        if not isinstance(metric, list):
            raise TypeError("metric must be list Type")

        for metc in metric:
            if metc not in cls_allowed_metrics:
                raise KeyError('metc: {} is not supported, '
                               '{} is can be supported'.format(metc, cls_allowed_metrics))

        targets = [value["cls_id"] for value in self.data_infos]
        # TODO: 为了支持多标签分类, 需要采用sigmoid函数, 这里没有想到好的统计. 后续再来实现吧!
        if use_sigmoid:
            assert thres_score is not None, "sigmoid must require a thres_score!"
        else:  # softmax模式
            return eval_cls_metrics(results, targets, metric, thres_score)