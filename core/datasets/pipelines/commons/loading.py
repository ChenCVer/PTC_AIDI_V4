# -*- coding:utf-8 -*-
import os.path as osp
import mmcv
import numpy as np
import os
import random
from bisect import bisect_right
from rraitools import ImageHelper, FileHelper
from ...registry import PIPELINES

try:
    from albumentations import Compose, BboxParams
except ImportError:
    Compose = None


__all__ = ['CustomLoadRefOfflineImageFromFile',
           'LoadImageFromFile',
           'LoadRefImageFromFile',
           'RandomRatioSelectSample',
           ]


def calc_cumulative_ratio(ratios):
    ratio_cumulative_list = [0]
    if np.sum(ratios) != 1.:
        ratios = ratios / np.sum(ratios)
    for i in range(len(ratios)):
        ratio_cumulative_list.append(ratio_cumulative_list[i] + ratios[i])
    return ratio_cumulative_list


def random_get_cls_id(ratio_cumulative_list):
    rand_num = random.random()  # 随机文件夹
    random_cls_id = bisect_right(ratio_cumulative_list, rand_num) - 1
    return random_cls_id


@PIPELINES.register_module
class CustomLoadRefOfflineImageFromFile(object):

    def __init__(self,
                 to_float32=False,
                 extensions=None):

        self.to_float32 = to_float32
        self.extensions = extensions

    def __call__(self, results):
        img_path = results["path"]
        img = results['img'].copy()
        # 从img_path出解析出图片后缀形式
        file_path, img_name = os.path.split(img_path)
        orig_img_extension = img_name[img_name.rfind("."):]
        ref_pth = img_path[:-len(orig_img_extension)] + self.extensions
        results['ref_path'] = ref_pth
        ref_img = ImageHelper.read_img(ref_pth)
        if self.to_float32:
            ref_img = ref_img.astype(np.float32)
        results['img'] = [img, ref_img]
        results['ori_shape'] = img.shape
        results['ref_ori_shape'] = ref_img.shape
        return results


@PIPELINES.register_module()
class LoadImageFromFile(object):

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk')):

        self.to_float32 = to_float32
        self.color_type = color_type
        # 默认是disk后端
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def __call__(self, results):

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if 'img_prefix' in results.keys() and results['img_prefix'] is not None:
            filename = osp.join(results['img_prefix'], results['img_info']['filename'])
        elif 'path' in results.keys() and results['path'] is not None:
            filename = results['path']
        else:
            filename = results['img_info']['filename']
        # 给定文件路径, 读取文件内容(字节流)
        img_bytes = self.file_client.get(filename)
        # 对字节内容进行解码, 读取出来的是: BGR格式
        img = mmcv.imfrombytes(img_bytes, flag=self.color_type)

        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename  # 全路径
        results['ori_filename'] = os.path.split(filename)[-1]  # 文件名
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['label'] = results.get('cls_id', None)
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]

        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)

        results['img_fields'] = ['img']  # 后续resize等需要用到的flag

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str


@PIPELINES.register_module
class LoadRefImageFromFile(object):

    def __init__(self,
                 root,
                 ref_path,
                 to_float32=False,
                 color_type='color',
                 extensions='.png',
                 exclude_extensions='_mask.png'):

        self.root = root
        self.ref_path = ref_path
        self.to_float32 = to_float32
        self.color_type = color_type
        self.extensions = extensions
        self.exclude_extensions = exclude_extensions

    def __call__(self, results):
        img_path = results['path']
        img = results['img'].copy()
        # 这个函数后续考虑开放出去, 因为随着任务类型的不同, 获取参考图的形式也不一样
        # 或者用户直接在外部自己重写LoadRefImageFromFile即可。
        ref_pth = self.get_ref_path(img_path)
        results['ref_path'] = ref_pth
        ref_img = ImageHelper.read_img(ref_pth)
        if self.to_float32:
            ref_img = ref_img.astype(np.float32)
        results['img'] = [img, ref_img]
        results['ori_shape'] = img.shape
        results['ref_ori_shape'] = ref_img.shape
        return results

    # TODO 不知道是否耗时，如果耗时，需要放在初始化里面
    def get_ref_path(self, img_path):
        if self.root.endswith('/'):
            temp_path = img_path.replace(self.root, '')
        else:
            temp_path = img_path.replace(self.root + '/', '')
        index_name = temp_path.split('/')
        if len(index_name) == 4:
            ref_path = os.path.join(self.ref_path, index_name[2])
        else:
            ref_path = self.ref_path
        ref_paths = FileHelper.get_file_path_list(ref_path, self.extensions, self.exclude_extensions)
        random_select_img_index = random.randint(0, len(ref_paths) - 1)
        ref_img_path = ref_paths[random_select_img_index]
        return ref_img_path

    def __repr__(self):
        return '{} (to_float32={}, color_type={})'.format(
            self.__class__.__name__, self.to_float32, self.color_type)


@PIPELINES.register_module
class RandomRatioSelectSample(object):
    """
    本方法主要为了实现针对分类/分割网络训练时,能精确控制每个batchsize中的类别比例.
    """
    def __init__(self,
                 class_list=None,
                 class_ratio_dict=None):

        self.class_list = class_list
        self.class_ratio_dict = class_ratio_dict
        # 这里是计算类别概率分布
        self._calc_cumulative_ratio()

    def _calc_cumulative_ratio(self):
        # 首先计算出大类别的累计概率分布(0.bg, 1.ng)
        self.classes_list = sorted(self.class_ratio_dict.keys())
        self.classes_ratios_list = [sum(self.class_ratio_dict[x]) for x in self.classes_list]
        self.classes_cumulative_ratio_list = calc_cumulative_ratio(self.classes_ratios_list)
        if self.class_list is not None:
            self.sub_class_ratio_dict = {}
            # 再计算每个类别中关于每一个小类的累计分布(0.bg下面的小文件夹的比例)
            for sub_class in self.classes_list:
                sub_class_ratio_list = calc_cumulative_ratio(self.class_ratio_dict[sub_class])
                self.sub_class_ratio_dict[sub_class] = sub_class_ratio_list

    def __call__(self, results):
        # 随机选择0.bg还是1.ng类别
        random_cls_id = random_get_cls_id(self.classes_cumulative_ratio_list)
        # 随机选择0.bg还是1.ng中小类的类别
        if self.class_list is not None:
            select_class_ratio = self.sub_class_ratio_dict[self.classes_list[random_cls_id]]
            random_sub_cls_id = random_get_cls_id(select_class_ratio)
            path = results[self.classes_list[random_cls_id]][self.class_list[random_sub_cls_id]]
        else:
            path = results[self.classes_list[random_cls_id]]

        # 随机获取一条路径
        path = random.choice(path)

        return path