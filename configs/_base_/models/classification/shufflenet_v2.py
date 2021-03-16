# model settings
model = dict(
    type='BaseClassifier',
    pretrained="https://download.openmmlab.com/mmclassification/v0/shufflenet_v2/shufflenet_v2_batch1024_imagenet_"
               "20200812-5bf4721e.pth",  # 预训练不能掉,否则收敛比较慢.
    backbone=dict(
        type='ShuffleNetV2',
        widen_factor=1.0,  # 0.5, 1.0, 1.5, 2.0
        out_indices=(3,),
        frozen_stages=-1,
        norm_eval=False,
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=5,
        # widen_factor:0.5/1.0/1.5 -> inchannel:1024
        # widen_factor: 2.0 -> inchannel:2048
        in_channels=1024,
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