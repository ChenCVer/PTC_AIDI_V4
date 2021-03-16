# model settings
model = dict(
    type='YoloV3',
    pretrained='open-mmlab://darknet53',
    # backbone
    backbone=dict(
        type='Darknet',
        depth=53,  # 网络深度
        out_indices=(3, 4, 5)),  # 多尺度输出.
    # neck
    neck=dict(
        type='YoloNeck',
        num_scales=3,  # 多尺度输出
        in_channels=[1024, 512, 256],
        out_channels=[512, 256, 128]),
    # head
    head=dict(
        type='YoloHead',
        num_classes=20,  # 类别.
        in_channels=[512, 256, 128],  # 与neck部分的out_channel对应.
        out_channels=[1024, 512, 256],  # 输出特征图是从小到大,故anchor size是从大到小
        anchors=[[116, 90], [156, 198], [373, 326],
                 [30, 61], [62, 45], [59, 119],
                 [10, 13], [16, 30], [33, 23]],
        anchors_mask=[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
        strides=[32, 16, 8],
        loss_level_weights=[1.0, 1.0, 1.0],
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
        # 回归损失
        loss_bbox_type="iou",
        loss_bbox=dict(
            type='GIoULoss',
            loss_weight=5.0,
            reduction="sum")
    )
)

# training and testing settings
# 该设置严格按照darknet比对进行
train_cfg = dict(
        pos_iou_thr=0.5,
        neg_iou_thr=0.3)

valid_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    conf_thr=0.005,
    nms=dict(type='nms', iou_thr=0.1),  # nms支持多种.
    max_per_img=100)

test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    conf_thr=0.005,
    nms=dict(type='nms', iou_thr=0.1),
    max_per_img=100)
