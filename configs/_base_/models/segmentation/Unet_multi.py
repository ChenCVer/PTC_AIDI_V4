layer_num = 4
root_channels = 16
kernel_size = 5

model = dict(
    type='BaseSegmentor',
    backbone=dict(
        type='UnetEncoder',
        input_channel=3,
        root_channels=root_channels,
        layer_num=layer_num,
        kernel_size=kernel_size,
        use_double_conv=True
    ),
    neck=None,
    head=dict(
        type='UnetHead',
        num_classes=2,
        root_channels=root_channels,
        layer_num=layer_num,
        kernel_size=kernel_size,
        use_double_conv=True,
        losser=dict(
            type='GeneralLosser',
            train_loss=[[dict(type='RRMultiDiceLoss'), dict(type='RRCrossEntropyLoss')]],
            train_weights=[[1, 1]],
            val_loss=[[dict(type='RRMultiDiceLoss'), dict(type='RRCrossEntropyLoss')]],
            val_weights=[[1, 1]],
        ),
        metricer=dict(type='SegMetricer')
    )
)

# model training and testing settings
train_cfg = None  # sigmoid模式(没有阈值)
valid_cfg = dict(num_classes=2, thres=0.5)
test_cfg = dict(num_classes=2, thres=0.5)
