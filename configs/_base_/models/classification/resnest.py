# model settings
model = dict(
    type='BaseClassifier',
    # TODO: 2021-02-23: 这组参数, 和预训练权重shape不一致.
    pretrained='open-mmlab://resnest50',
    backbone=dict(
        type='ResNeSt',
        in_channels=3,
        depth=50,  # 50, 101, 152
        reduction_factor=4,
        groups=32,
        width_per_group=4,
        radix=2,
        avg_down_stride=True,
        num_stages=4,
        out_indices=(3,),
        style='pytorch',
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