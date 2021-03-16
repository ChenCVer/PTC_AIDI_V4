from abc import ABC

from ..registry import MODELS
from ..base.detector import BaseDetector

__all__ = ["YoloV3"]


@MODELS.register_module()
class YoloV3(BaseDetector, ABC):

    def __init__(self,
                 backbone,
                 neck,
                 head,
                 train_cfg=None,
                 valid_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(YoloV3, self).__init__(backbone,
                                     neck,
                                     head,
                                     train_cfg,
                                     valid_cfg,
                                     test_cfg,
                                     pretrained)
