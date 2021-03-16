# -*- coding:utf-8 -*-
import collections
import random
import cv2
from rraitools import build_from_cfg
from core.datasets.registry import PIPELINES


__all__ = ['Transform',
           'Compose']

# todo: bbox可能有bug, 因为, gt_bbox和gt_label是分开的, 因此, 在做比如
#  RandomShift操作时可能gt_bboxes和gt_labels可能不同步.
LABEL_KEY = {0: 'gt_semantic_seg', 1: 'gt_bboxes', 2: 'keypoint_label'}


@PIPELINES.register_module
class Compose(object):
    """Compose multiple transforms sequentially.

    Args:
        transforms (Sequence[dict | callable]): Sequence of transform object or
            config dict to be composed.
    """

    def __init__(self, transforms):
        assert isinstance(transforms, collections.abc.Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict')

    def __call__(self, data):
        """Call function to apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
           dict: Transformed data.
        """
        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class Transform(object):
    def __init__(self, mode, prob):
        self.probability = prob
        self.mode = mode
        if mode == 0:  # seg
            self._transform_fun = self.apply_segmentation
        elif mode == 1:  # det
            self._transform_fun = self.apply_box
        elif mode == 2:  # keypoint
            self._transform_fun = self.apply_coords
        elif mode == 100:  # 多种数据类型同时增强模式
            self._transform_fun = [self.apply_segmentation, self.apply_box, self.apply_coords]
        else:
            raise Exception('Only supports 0=segmentation,1=box,2=coords mode')
        self.parameter_dict = {}

    def __call__(self, results, **kwargs):
        image = results.get('img', None)
        assert image is not None
        if random.random() < self.probability:
            is_not_iter = False
            if not isinstance(image, list):
                image = [image]
                is_not_iter = True
            img_output_list = []
            # 对img进行操作.
            for index, img in enumerate(image):
                if index == 0:
                    self.get_parameter(img, **kwargs)
                img_one = self.apply_image(img, **kwargs)
                img_output_list.append(img_one)
            if is_not_iter:
                img_output_list = img_output_list[0]
            results['img'] = img_output_list

            # 对label进行操作.
            if self.mode != 100:
                results = self._apply_label(results, self.mode, self._transform_fun, **kwargs)
            else:
                for i, transfrom_fun in enumerate(self._transform_fun):
                    results = self._apply_label(results, i, transfrom_fun, **kwargs)
            return results
        else:
            return results

    def _apply_label(self, results, mode, transform_fun, **kwargs):
        label = results.get(LABEL_KEY[mode], None)
        if label is not None:
            label_output_list = []
            if isinstance(label, list):
                for lab in label:
                    label_one = transform_fun(lab, **kwargs)
                    label_output_list.append(label_one)
                results[LABEL_KEY[self.mode]] = label_output_list
            else:
                label = transform_fun(label, **kwargs)
                results[LABEL_KEY[self.mode]] = label
        return results

    def get_parameter(self, image, **kwargs):
        """
        get augmentation parameter, save paremters to apply on multiple objects.
        """
        pass

    def apply_image(self, image, interpolation=cv2.INTER_LINEAR, **kwargs):
        """
        execute augmentation on images.
        """

        return image

    def apply_segmentation(self, mask, **kwargs):
        """
        execute augmentation on mask.
        """
        return mask

    def apply_box(self, box_xyxy, **kwargs):
        """
        execute augmentation on bbox.
        """
        return box_xyxy

    def apply_coords(self, coords_nxy, **kwargs):
        """
        execute augmentation on coordinates.
        """
        return coords_nxy