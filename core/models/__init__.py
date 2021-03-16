from .registry import (LOSSES, BACKBONES, HEADS, NECKS, MODELS)

from .builder import (build_backbone, build_head, build_loss,
                      build_neck, build_model)

# Notes: from .registry import * 和 from .builder import * 必须放在from .base import *
# 的前面, 因为python是顺序执行程序.而from .base import * 等里面的代码有 from core.models import
# BACKBONES等, 所以必须models的__init__.py文件首先执行from .registry import *等.

from .base import *
from .classification import *
from .components import *
from .objectdetection import *
from .segementation import *