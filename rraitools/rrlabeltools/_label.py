# -*- coding: utf-8 -*-
import pprint
import os
import numpy as np
import cv2


def assert_point(point):
    p = np.array(point, dtype=float)
    assert p.shape == (2,)
    return p


class BoundingBox(object):
    def __init__(self, p1, p2, class_name):
        assert isinstance(class_name, str)
        p1 = assert_point(p1)
        p2 = assert_point(p2)

        self.ul_point = np.array([min(p1[0], p2[0]), min(p1[1], p2[1])], dtype=int)  # 左上角 upper left -> ul
        self.lr_point = np.array([max(p1[0], p2[0]), max(p1[1], p2[1])], dtype=int)  # 右下角 lower right -> lr
        self.width = self.lr_point[0] - self.ul_point[0]
        self.height = self.lr_point[1] - self.ul_point[1]
        self.class_name = class_name
        self.clip_coord()

    def __str__(self):
        return '{0}(\n{1}\n)'.format(self.__class__.__name__, pprint.pformat(self.to_dict()))

    def __repr__(self):
        return self.__str__()

    def to_dict(self):
        return {
            'upper_left': self.ul_point,
            'lower_right': self.lr_point,
            'width': self.width,
            'height': self.height,
            'class': self.class_name
        }

    def is_in_box(self, points):
        # 检查关键点是否在框内
        for p in points:
            if not (self.ul_point[0] < p[0] < self.lr_point[0] and self.ul_point[1] < p[1] < self.lr_point[1]):
                return False
        return True

    def __add__(self, other):
        other = assert_point(other)
        p1 = self.ul_point + other
        p2 = self.lr_point + other
        return self.__class__(p1, p2, self.class_name)

    def __sub__(self, other):
        other = assert_point(other)
        p1 = self.ul_point - other
        p2 = self.lr_point - other
        return self.__class__(p1, p2, self.class_name)

    def clip_coord(self):
        self.ul_point = np.clip(self.ul_point, 0, 9999)
        self.lr_point = np.clip(self.lr_point, 0, 9999)

class KeyPoints(object):
    def __init__(self, points_nx2):
        points = np.array(points_nx2, dtype=float)
        assert len(points.shape) == 2
        assert points.shape[1] == 2

        self.points = points

    def __len__(self):
        return len(self.points)

    def __getitem__(self, item):
        return self.points[item]

    def __iter__(self):
        return self.points.__iter__()

    def __str__(self):
        return '{0}(\n{1}\n)'.format(self.__class__.__name__, pprint.pformat(self.points))

    def __repr__(self):
        return self.__str__()

    def __add__(self, other):
        other = assert_point(other)
        return self.__class__(self.points + other)

    def __sub__(self, other):
        other = assert_point(other)
        return self.__class__(self.points - other)


class SampleLabel(object):
    def __init__(self, bounding_box, key_points):
        assert isinstance(bounding_box, BoundingBox)
        assert isinstance(key_points, KeyPoints)

        self._bounding_box = bounding_box
        self._key_points = key_points

    @property
    def class_name(self):
        return self._bounding_box.class_name

    @property
    def bbox(self):
        return self._bounding_box

    @property
    def key_points(self):
        return self._key_points

    def __str__(self):
        return '{0}(\n{1}\n)'.format(self.__class__.__name__, pprint.pformat(self.to_dict()))

    def __repr__(self):
        return self.__str__()

    def to_dict(self):
        # 转换为字典
        return {
            'bounding_box': self._bounding_box,
            'key_points': self._key_points,
            'class': self.class_name
        }


class ImageLabel(object):
    def __init__(self):
        self._image_path = ''
        self._label_path = ''

        self._all_bounding_box = []
        self._all_key_points = []
        self._unrecognized_labels = []

        self._matched_samples = []
        self._lonely_bboxes = []
        self._lonely_kpts = []

    @property
    def image_path(self):
        return self._image_path

    @property
    def label_path(self):
        return self._label_path

    @image_path.setter
    def image_path(self, value):
        self._image_path = value

    @label_path.setter
    def label_path(self, value):
        self._label_path = value

    @property
    def bboxes(self):
        return self._all_bounding_box

    @property
    def samples(self):
        return self._matched_samples

    @property
    def image_shape(self):
        image = cv2.imread(self.image_path)
        height, width, _ = image.shape
        shape = (height, width)
        return shape

    def to_relpath(self, root_dir):
        self._image_path = os.path.relpath(self._image_path, root_dir)
        self._label_path = os.path.relpath(self._label_path, root_dir)

    def __len__(self):
        # 返回sample个数
        return len(self._matched_samples)

    def __str__(self):
        return '{0}(\n{1}\n)'.format(self.__class__.__name__, pprint.pformat(self.to_dict()))

    def __repr__(self):
        return self.__str__()

    def __iter__(self):
        self._matched_samples.__iter__()

    def __getitem__(self, item):
        return self._matched_samples[item]

    def add(self, label_object):
        if isinstance(label_object, BoundingBox):
            self._all_bounding_box.append(label_object)
        elif isinstance(label_object, KeyPoints):
            self._all_key_points.append(label_object)
        else:
            raise ValueError

    def dump(self, label_data):
        self._unrecognized_labels.append(label_data)

    def append(self, sample_label):
        assert isinstance(sample_label, SampleLabel)
        self._matched_samples.append(sample_label)
        self._all_bounding_box.append(sample_label.bbox)
        self._all_key_points.append(sample_label.key_points)

    def register_key_points(self):
        """
        以关键点是否在bbox内为标准，匹配bbox和关键点，并组成SampleLabel。
        返回SampleLabel的列表，以及未匹配成功的bbox和关键点。
        """
        lonely_kpts = self._all_key_points.copy()
        all_bounding_box = self._all_bounding_box.copy()

        for key_points in self._all_key_points:
            for bounding_box in all_bounding_box:
                if bounding_box.is_in_box(key_points):
                    self._matched_samples.append(SampleLabel(bounding_box, key_points))
                    all_bounding_box.remove(bounding_box)
                    lonely_kpts.remove(key_points)
                    break
        self._lonely_bboxes = all_bounding_box
        self._lonely_kpts = lonely_kpts

    def list_nkpts(self):
        # 以列表形式返回图中所有样本的关键点数量
        return [len(sample.key_points) for sample in self._matched_samples]

    def list_class_names(self):
        return [sample.class_name for sample in self._matched_samples]

    def remove_samples_by_nkpts(self, nkpts_to_keep):
        self._matched_samples = list(filter(lambda x: len(x.key_points) == nkpts_to_keep, self._matched_samples))

    def to_dict(self):
        # 转换为字典
        return {
            'image_path': self._image_path,
            'label_path': self._label_path,
            'samples': self._matched_samples
        }
