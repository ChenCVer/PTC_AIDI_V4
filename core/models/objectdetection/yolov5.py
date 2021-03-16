from abc import ABC

from ..registry import MODELS
from ..base.detector import BaseDetector

__all__ = ["YoloV5"]


@MODELS.register_module()
class YoloV5(BaseDetector, ABC):

    def __init__(self,
                 depth_multiple,
                 width_multiple,
                 backbone,
                 neck=None,
                 head=None,
                 train_cfg=None,
                 valid_cfg=None,
                 test_cfg=None,
                 pretrained=None):

        backbone['depth_multiple'] = depth_multiple
        backbone['width_multiple'] = width_multiple

        if neck is not None:
            neck['depth_multiple'] = depth_multiple
            neck['width_multiple'] = width_multiple

        if head is not None:
            head['depth_multiple'] = depth_multiple
            head['width_multiple'] = width_multiple

        super(YoloV5, self).__init__(backbone,
                                     neck,
                                     head,
                                     train_cfg,
                                     valid_cfg,
                                     test_cfg,
                                     pretrained)
