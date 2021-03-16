# model settings
model = dict(
    type='BaseClassifier',
    pretrained='https://download.openmmlab.com/mmclassification/v0/se-resnet/se-resnet50_batch256_imagenet_20200804'
               '-ae206104.pth',  # 预训练不能掉,否则收敛比较慢.
    backbone=dict(
        type='SEResNet',
        in_channels=3,  # 网络输入通道(如果是单通道灰度图, 请变为3通道), 如果含有参考图, 则需要变为6.
        depth=50,  # 50, 101, 152
        num_stages=4,
        out_indices=(3,),
        style='pytorch',
        norm_eval=False,
        zero_init_residual=False,),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=5,     # 类别数
        in_channels=2048,  # depth:50/101/152->in_channel2048;
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
