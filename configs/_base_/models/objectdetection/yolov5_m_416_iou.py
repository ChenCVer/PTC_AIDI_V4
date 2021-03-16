# YOLO model
model = dict(
    type='YoloV5',
    pretrained=None,
    depth_multiple=1,  # model depth multiple 类似efficient net 代表深度   l:1 m:0.67 s:0.33 x:1.33
    width_multiple=1,  # layer channel multiple 类似efficient net 代表宽度 卷积的输出通道数 l:1 m:0.75 s:0.50 x:1.25
    # backbone
    backbone=dict(
        type='YOLOv5Darknet',
        # 其中focus[0]表示输出通道数, focus[1]表示focus中的conv的kernel_size, focus[2]表示conv中的stride
        focus=[64, 3, 1],
        in_channels=3,
        bottle_depths=[3, 9, 9, 3],  # CSP模块中Res_Unit的个数.
        out_channels=[128, 256, 512, 1024],
        shortcut=[True, True, True, False],
        out_indices=(2, 3, 4,),
        spp=[5, 9, 13],  # spp 核大小
        norm_cfg=dict(type='BN', requires_grad=True),
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1)
    ),
    # neck
    neck=dict(
        type='Yolov5Neck',
        in_channels=1024,
        upsampling_mode='nearest',
        out_channels=[512, 256, 256, 512],
        bottle_depths=[3, 3, 3, 3],
        shortcut=[False, False, False, False],
        norm_cfg=dict(type='BN', requires_grad=True),
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1)
    ),
    # head
    head=dict(
        type='Yolov5Head',
        num_classes=20,
        in_channels=[1024, 512, 256],
        label_smooth=False,  # 是否启用label_smooth
        deta=0.01,  # label_smooth的deta值
        anchors=[[116, 90], [156, 198], [373, 326],
                 [30, 61], [62, 45], [59, 119],
                 [10, 13], [16, 30], [33, 23]],
        anchors_mask=[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
        strides=[32, 16, 8],
        nms_type='nms',  # Support types :nms or soft_nms
        nms_thr=.5,  # nms 阈值
        ignore_thre=.4,  # yolo原系列 置信度损失 负样本 neg_loss忽略部分的阈值
        conf_balances=[4, 1, 0.4],  # 对不同尺度的置信度权重设置[小,中,大]
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
            reduction='sum'),
        loss_conf=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
            reduction='sum'),
        loss_bbox=dict(
            type='GIoULoss',
            loss_weight=10.0),
    )
)

# training and testing settings
train_cfg = None
val_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    conf_thr=0.005,
    nms=dict(type='nms', iou_thr=0.45),  # nms支持多种.
    max_per_img=100)

test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    conf_thr=0.005,
    nms=dict(type='nms', iou_thr=0.45),
    max_per_img=100)
