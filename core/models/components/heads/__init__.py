from .cls import *
from .seg import *
from .det import *
from .keypoints import *

__all__ = [
    'LinearClsHead',
    'UnetHead',
    'YoloHead',
    'Yolov5Head',
    'CenternetHead_hm',
    'UnetHead_Hm'
]