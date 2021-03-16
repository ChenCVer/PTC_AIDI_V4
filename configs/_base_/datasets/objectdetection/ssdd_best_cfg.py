# dataset settings
dataset_type = 'VOCDataset'
data_root = '/home/cxj/Desktop/data/ssdd_voc/'
img_endswith = ".jpg"
CLASSES = ('ship',)
img_scale = (416, 416)
# LoadImageFromFile函数实现导入的图像格式是BGR.
img_norm_cfg = dict(
    mean_bgr=[0.0, 0.0, 0.0],
    std_bgr=[255.0, 255.0, 255.0],
    to_rgb=False)

train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=False),  # 读取出来的格式是BGR
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    # dict(type='Mosaic', img_scale=img_scale[0], maxlen=2000, mosaic_ratio=0.5,
    #      transforms=[
    #          dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    #      ]),
    # dict(type='Resize', img_scale=[(608, 608), (512, 512), (416, 416)],
    #      keep_ratio=True, multiscale_mode='value'),
    # dict(type='RandomFlip', flip_ratio=0.6,
    #      direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            # dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=0,
    eval_samples_per_gpu=1,
    eval_workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007/ImageSets/Main/train.txt',
        img_prefix=data_root + 'VOC2007/',
        pipeline=train_pipeline,
        extensions=img_endswith,
        classes=CLASSES),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
        img_prefix=data_root + 'VOC2007/',
        pipeline=val_pipeline,
        extensions=img_endswith,
        classes=CLASSES),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
        img_prefix=data_root + "VOC2007/",
        pipeline=test_pipeline,
        extensions=img_endswith,
        classes=CLASSES))
