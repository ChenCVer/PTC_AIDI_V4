"""
MobileNetV1:
可以看成在VGG的基础上将卷积层换成深度可分离卷积,激活函数换成ReLU6, 其基本单元为:
DW_CONV_3*3->ReLU6->PW_CONV_1*1.
MobileNetV2:
V1版本存在问题: DW_CONV_3*3部分几乎为稀疏矩阵(很多位置为0.), 通过分析认为是:
对于一个m维度空间, 将其升维到低维度和高维度空间后进行ReLU运算, 然后将其结果降维m
维度对比于原输入差距, 相比于高维度空间, 其在低维度空间下进行的ReLU运算信息丢失大.
V2解决办法: 由于DW_CONV没有改变通道的能力, 因此, 如果继续DW_CONV+ReLU6的形式肯定
不行, 因此, 可以在DW深度卷积之前使用PW卷积进行升维(升维倍数为t,t=6),再在一个更高维
的空间中进行卷积操作来提取特征, 因此, 其Unit变为:
input->PW_CONV_1*1->ReLU6->DW_CONV_3*3->ReLU6->PW_CONV_1*1->out, 同时,
V2也借鉴了残差思想, 在input和out之间进行shortcut链接.
"""
# model settings
model = dict(
    type='BaseClassifier',
    pretrained=None,
    backbone=dict(
        type='MobileNetV2',
        widen_factor=1.0),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=5,
        in_channels=1280,
        loss=dict(
            type='GeneralLosser',
            losses=[dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0), ],
        ),
    ),
)

train_cfg = None
valid_cfg = None
test_cfg = dict(score_thr=0.8, use_sigmoid=False, out_cam=True)