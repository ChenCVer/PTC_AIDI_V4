# model settings
model = dict(
    type='BaseClassifier',
    pretrained="https://download.openmmlab.com/mmclassification/v0/shufflenet_v1/shufflenet_v1_batch1024_imagenet_"
               "20200804-5d6cec73.pth",  # 预训练不能掉,否则收敛比较慢.
    backbone=dict(
        type='ShuffleNetV1',
        groups=3,  # 1, 2, 3, 4, 8
        widen_factor=1.0,
        out_indices=(2,),
        frozen_stages=-1,
        norm_eval=False,
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=5,
        in_channels=960,  # 576, 800, 960, 1088, 1536
        loss=dict(
            type='GeneralLosser',
            losses=[
                # dict(type='FocalLoss', use_sigmoid=False, loss_weight=1.0),
                dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
            ],
        ),
    )
)


train_cfg = None
valid_cfg = None
test_cfg = dict(score_thr=0.8, use_sigmoid=False, out_cam=True)