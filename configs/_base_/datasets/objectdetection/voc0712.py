# dataset settings
dataset_type = 'VOCDataset'
data_root = '/home/georg/Documents/VOC_mini/VOCdevkit/'
img_endswith = ".jpg"
# LoadImageFromFile函数实现导入的图像格式是BGR.
img_norm_cfg = dict(
    mean_bgr=[123.675, 116.28, 103.53],
    std_bgr=[58.395, 57.12, 57.375], to_rgb=False)
img_scale = (512, 512)

train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=False),  # 读取出来的格式是BGR
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    # dict(type='Mosaic', img_scale=512, maxlen=2000, mosaic_ratio=0.5,
    #      transforms=[
    #          dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
    #      ]),
    dict(type='MinIoURandomCrop', min_ious=(0.7, 0.8, 0.9), min_crop_size=0.7),
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
    samples_per_gpu=4,
    workers_per_gpu=0,
    eval_samples_per_gpu=1,
    eval_workers_per_gpu=0,
    train=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'VOC2007/ImageSets/Main/train.txt',
            img_prefix=data_root + 'VOC2007/',
            extensions=img_endswith,
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007/ImageSets/Main/val.txt',
        img_prefix=data_root + 'VOC2007/',
        extensions=img_endswith,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
        img_prefix=data_root + 'VOC2007/',
        extensions=img_endswith,
        pipeline=test_pipeline))
