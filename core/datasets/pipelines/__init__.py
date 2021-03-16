# -*- coding:utf-8 -*-
from .classification import *
from .commons import *
from .objectdetection import *
from .segmentation import *
from .keypoints import *
from .utils import *


"""
Notes: 如果你需要自定义img的loading, transforms和formating, 则只需要在对应的任务(cls, seg和det)
文件夹下的对应文件夹中实现, 然后用装饰器装饰上OK了, 如果不需要自定义, 则用common文件夹中的就好.
"""