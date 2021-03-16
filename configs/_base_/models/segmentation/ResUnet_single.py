layer_num = 4
root_channels = 16
kernel_size = 3
use_double_conv = True
shortcut = True

model = dict(
    type='BaseSegmentor',
    backbone=dict(
        type='UnetEncoder',
        input_channel=3,
        root_channels=root_channels,
        layer_num=layer_num,
        kernel_size=kernel_size,
        use_double_conv=use_double_conv,
        shortcut=shortcut,  # backbone 采用残差链接.
        dowmsample_style='maxpool',  # 下采样方式: ['maxpool', 'conv_down']
    ),
    neck=None,
    head=dict(
        type='UnetHead',
        num_classes=1,
        root_channels=root_channels,
        layer_num=layer_num,
        kernel_size=kernel_size,
        use_double_conv=use_double_conv,
        shortcut=shortcut,
        loss=dict(
            type='GeneralLosser',
            train_loss=[[dict(type='RRBinaryDiceLoss'), dict(type='RRBCELoss')]],
            train_weights=[[5, 1]],
            val_loss=[[dict(type='RRBinaryDiceLoss'), dict(type='RRBCELoss')]],
            val_weights=[[5, 1]],
        ),
    )
)

# model training and testing settings
train_cfg = None  # sigmoid模式(没有阈值)
valid_cfg = dict(num_classes=1, score_thr=0.01, area_thr=100)
# 整图测试
test_cfg = dict(
    type="whole_img_mode",
    num_classes=1,
    score_thr=None,
    area_thr=100)
# 小图测试
# test_cfg = dict(
#     type="piece_img_mode",
#     num_classes=1,
#     score_thr=0.5,
#     area_thr=100,
#     piece_shape=(256, 256),
#     overlap_hw=(64, 64))