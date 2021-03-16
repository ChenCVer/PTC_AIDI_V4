# model settings
model = dict(
    type='BaseKeyPointer',
    pretrained='torchvision://resnet18',
    backbone=dict(
        type='ResNet',
        in_channels=3,
        depth=18,  # 18, 34, 50, 101, 152
        num_stages=4,
        deep_stem=False,
        out_indices=(3,),
        style='pytorch',
        norm_eval=False,
        zero_init_residual=False,
    ),
    neck=None,
    head=dict(
        type='CenternetHead_hm',
        num_classes=1,
        in_channels=512,  # depth:18/34->in_channel:512; depth:50/101/2048->in_channel2048;
        losser=dict(
            type='GeneralLosser',
            train_loss=[dict(type='MSELoss'), ],
            val_loss=[dict(type='MSELoss'), ]),
        metricer=dict(type='KeyPointMetricer', down_ratio=4)
    )
)

train_cfg = None
valid_cfg = None
test_cfg = None
