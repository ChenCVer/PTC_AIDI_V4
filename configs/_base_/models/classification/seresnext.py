"""
TODO: 2021-02-25: 开始从seresnet实现算法.
"""
# model settings
model = dict(
    type='BaseClassifier',
    pretrained='https://download.openmmlab.com/mmclassification/v0/se-resnet/se-resnet50_batch256_imagenet_20200804'
               '-ae206104.pth',
    backbone=dict(
        type='SEResNeXt',
        in_channels=3,
        depth=50,  # 50, 101, 152
        num_stages=4,
        groups=32,
        width_per_group=4,
        deep_stem=False,
        out_indices=(3,),
        style='pytorch',
        se_ratio=16,
        norm_eval=False,
        zero_init_residual=False,
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=5,
        in_channels=2048,  # depth:50/101/2048->in_channel2048;
        loss=dict(
            type='GeneralLosser',
            losses=[dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0), ],
        ),
    )
)

train_cfg = None
valid_cfg = None
test_cfg = dict(score_thr=0.8, use_sigmoid=False, out_cam=True)
