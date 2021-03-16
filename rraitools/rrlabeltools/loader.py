# -*- coding: utf-8 -*-
import os
import json
from ._label import BoundingBox, KeyPoints, ImageLabel


class LabelLoader(object):
    """
    用于加载一份文件中的标注。
    需要通过继承此类来实现对于不同标注软件的支持。
    """
    def __init__(self):
        self._load_label_factory = self._build_factory()

    def load(self, file_path):
        image_label = None
        raw_data = self._load_label_file(file_path)
        if raw_data is not None and self._is_data_good(raw_data):
            image_label = self._traverse_data(raw_data)
            image_label.register_key_points()
        return image_label

    @staticmethod
    def _load_label_file(file_path):
        """
        读取label文件。不同标注软件生成的label文件可能不一样，所以需要继承。
        返回标注数据，格式取决于标注软件。
        :param file_path:
        :return:
        """
        raise NotImplementedError

    @staticmethod
    def _is_data_good(data):
        raise NotImplementedError

    def _traverse_data(self, data):
        raise NotImplementedError

    def _build_factory(self):
        raise NotImplementedError

    @staticmethod
    def _load_rectangle_label(data):
        raise NotImplementedError

    @staticmethod
    def _load_point_label(data):
        raise NotImplementedError

    @staticmethod
    def _load_linestrip_label(data):
        raise NotImplementedError

    @staticmethod
    def _load_polygon_label(data):
        raise NotImplementedError


class LabelmeLoader(LabelLoader):
    FILENAME_SUFFIX = 'json'

    def __init__(self):
        super().__init__()

    @staticmethod
    def _load_rectangle_label(data):
        pass

    @staticmethod
    def _load_point_label(data):
        pass

    @staticmethod
    def _load_linestrip_label(data):
        pass

    @staticmethod
    def _load_polygon_label(data):
        pass


class LabelRRLoader(LabelLoader):
    FILENAME_SUFFIX = 'json'

    def __init__(self):
        super().__init__()

    @staticmethod
    def _load_label_file(file_path):
        try:
            data = json.load(open(file_path))
        except:
            data = None
        return data

    @staticmethod
    def _is_data_good(data):
        # 目前无法判断
        return True

    def _traverse_data(self, data):
        image_label = ImageLabel()
        for d in data:
            label_type = d[0]
            if label_type in self._load_label_factory.keys():
                label = self._load_label_factory[label_type](d)
                image_label.add(label)  # 有用的标注
            else:
                image_label.dump(d)  # 没用的标注
        return image_label

    def _build_factory(self):
        return {
            'rectangle': self._load_rectangle_label,
            'point': self._load_point_label,
            'broken_line': self._load_linestrip_label,
            'polygon_no_filled': self._load_polygon_label
        }

    @staticmethod
    def _load_rectangle_label(data):
        rectangle = data[1]
        class_name = data[4]
        p1 = rectangle[0]
        p2 = rectangle[1]
        return BoundingBox(p1, p2, class_name)

    @staticmethod
    def _load_point_label(data):
        points = data[1]
        return KeyPoints(points)

    @staticmethod
    def _load_linestrip_label(data):
        points = data[1]
        return KeyPoints(points)

    @staticmethod
    def _load_polygon_label(data):
        points = data[1]
        return KeyPoints(points)


def get_label_loader(tool):
    if tool == 'Labelme':
        return LabelmeLoader
    elif tool == 'LabelRR':
        return LabelRRLoader
    else:
        raise ValueError


class BatchLoader(object):

    def __init__(self, tool, image_type='png'):
        self.loader = get_label_loader(tool)()
        self._image_type = image_type

    def load(self, root_dir):
        return self._traverse_dir(root_dir)

    def _traverse_dir(self, root_dir):
        """
        遍历root_dir下所有标注文件，如果能找到与标注文件所对应的图片，那么就为这个标注文件生成一个ImageLabel对象。
        如果找不到对应的图片，则无视这个标注文件。
        在初始化ImageLabel对象时，使用图片和标注文件的相对路径。
        返回一个由ImageLabel对象所组成的列表。
        :param root_dir:
        :return:
        """

        items = sorted(os.listdir(root_dir))  # 遍历当前文件夹
        items = [os.path.join(root_dir, item) for item in items]  # 获得当前文件夹内所有项目的完整路径
        items = filter(lambda x: os.path.isdir(x) or x.endswith(self.loader.FILENAME_SUFFIX), items)  # 去除非文件夹及非标注文件

        image_labels = []
        for item in items:
            if os.path.isdir(item):
                # 递归处理文件夹
                image_labels.extend(self._traverse_dir(item))
            else:
                # 处理标注文件
                parent_dir, label_filename = os.path.split(item)
                image_filename = '.'.join([label_filename.split('.')[0], self._image_type])
                image_path = os.path.join(parent_dir, image_filename)  # 获取与标注文件同名的图片文件路径
                if os.path.exists(image_path):
                    # 如果图片存在，则处理标注，否则无视这个文件
                    image_label = self.loader.load(item)
                    if image_label is not None:
                        image_label.image_path = image_path
                        image_label.label_path = item
                        image_labels.append(image_label)

        return image_labels
