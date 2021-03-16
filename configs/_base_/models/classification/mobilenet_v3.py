"""
TODO: 2021-02-24: MobileNetV3, 还需要抽时间具体看看, 研究下.
"""
# model settings
model = dict(
    type='BaseClassifier',
    backbone=dict(
        type='MobileNetV3',
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