import torch
from core.models import ClsResNet
from core.models import ResNet

inputs = torch.ones(10, 3, 32, 32) * 100.0
seed = 0
torch.manual_seed(seed)            # 为CPU设置随机种子
torch.cuda.manual_seed(seed)       # 为当前GPU设置随机种子

# 分类resnet
clsresnet = ClsResNet(depth=18, out_indices=(3,))
clsresnet.train()
clsoutput = clsresnet.forward(inputs)
print("clsoutput[0] = ", clsoutput[0][0, :10, ...])


# 检测resnet
resnet = ResNet(depth=18, out_indices=(3,), norm_eval=False)
resnet.train()
detoutput = resnet.forward(inputs)
print("output[0] = ", detoutput[0][0, :10, ...])


print(torch.equal(clsoutput[0], detoutput[0]))