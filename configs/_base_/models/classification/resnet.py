# model settings
model = dict(
    type='BaseClassifier',
    pretrained='torchvision://resnet18',  # 预训练不能掉,否则收敛比较慢.
    backbone=dict(
        type='ResNet',
        in_channels=3,  # 网络输入通道(如果是单通道灰度图, 请变为3通道), 如果含有参考图, 则需要变为6.
        depth=18,  # 18, 34, 50, 101, 152
        num_stages=4,
        deep_stem=True,  # Replace 7x7 conv in input stem with 3 3x3 conv, it's useful when the input size is small.
        out_indices=(3,),
        style='pytorch',
        # TODO: 由于需要兼容, norm_eval=False, 则分类网络在val阶段的BN模式仍然有效.
        #  因此, val阶段下, 其BN仍然为True.导致统计结果不准确！
        norm_eval=False,  # norm_eval这个必须要关闭.否则backbone在train阶段BN是eval模式.
        # 将BN层的γ初始化0: 对于ResNet来说, 可以将残差块后面接的batch normalization层
        # 的y=γx+β中的γ初始化为0. 这样做的好处是,在初始化阶段可以简化网络, 更快训练.
        # 参考: https://zhuanlan.zhihu.com/p/66393448
        zero_init_residual=True,
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=5,  # 类别数
        in_channels=512,  # depth:18/34->in_channel:512; depth:50/101/2048->in_channel2048;
        loss=dict(
            type='GeneralLosser',
            losses=[
                # dict(type='FocalLoss', use_sigmoid=False, loss_weight=1.0),
                dict(type="LabelSmoothLoss", label_smooth_eps=0.1, num_classes=5),
                    ],
        ),
    )
)

train_cfg = None
valid_cfg = None
test_cfg = dict(score_thr=0.8, use_sigmoid=False, out_cam=True)
