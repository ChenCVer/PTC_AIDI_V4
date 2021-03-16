# todo: 2021-02-25 这里需要编写一个利用torchstat计算model的复杂度指标的工具
from torchstat import stat
import torchvision.models as models

model = models.resnet34()
stat(model, (3, 224, 224))