# Copyright (c) Open-MMLab. All rights reserved.
from abc import ABCMeta, abstractmethod


# 继承ABCMeta元类, 使其无法直接实例化
class BaseFileHandler(metaclass=ABCMeta):

    @abstractmethod
    def load_from_fileobj(self, file, **kwargs):
        # 文件读取
        pass

    @abstractmethod
    def dump_to_fileobj(self, obj, file, **kwargs):
        # 文件存储, 需要传入对象obj和file
        pass

    @abstractmethod
    def dump_to_str(self, obj, **kwargs):
        # dump成字符串返回, 当你不想保存时候使用
        pass

    # 对外实际上是采用下面两个api, 继承BaseFileHandler的类只需要重写上述三个方法.
    def load_from_path(self, filepath, mode='r', **kwargs):
        with open(filepath, mode) as f:
            return self.load_from_fileobj(f, **kwargs)

    def dump_to_path(self, obj, filepath, mode='w', **kwargs):
        with open(filepath, mode) as f:
            self.dump_to_fileobj(obj, f, **kwargs)
