layer_num = 4
root_channels = 16
kernel_size = 3
use_double_conv = True

model = dict(
    type='BaseSegmentor',
    backbone=dict(
        type='UnetEncoder',
        input_channel=3,
        root_channels=root_channels,
        layer_num=layer_num,
        kernel_size=kernel_size,
        use_double_conv=use_double_conv
    ),
    neck=None,
    head=dict(
        type='UnetHead',
        num_classes=1,
        root_channels=root_channels,
        layer_num=layer_num,
        kernel_size=kernel_size,
        use_double_conv=use_double_conv,
        loss=dict(
            type='GeneralLosser',
            losses=[dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
                    dict(type='BinaryDiceLoss', loss_weight=4.0),
                    ],
        ),
    )
)

# model training and testing settings
train_cfg = None  # sigmoid模式(没有阈值)
valid_cfg = dict(num_classes=1, score_thr=None, area_thr=10)
# 整图测试
# test_cfg = dict(
#     type="whole_img_mode",
#     num_classes=1,
#     score_thr=None,
#     area_thr=5)
# 小图测试
test_cfg = dict(
    type="piece_img_mode",
    num_classes=1,
    score_thr=0.9,
    area_thr=10,
    piece_shape=(512, 256),
    overlap_hw=(64, 64))
