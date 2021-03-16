import collections

dataset_type = 'SegDataset'
data_root = '/home/cxj/Desktop/data/electronic_datasets/electronic_seg_line/'
img_norm_cfg = dict(mean_bgr=[0.0, 0.0, 0.0],
                    std_bgr=[255.0, 255.0, 255.0], to_rgb=False)
# ********************************************************************************** #
# 离线训练不需要提供颜色列表信息
_class_ratios_dict = {"0.bg": [0],
                      "1.ng": [1]}
class_order_dict = collections.OrderedDict({
    "0.bg": (0, 0, 0),
    "1.ng": (255, 255, 255)})
num_classes = len(class_order_dict) - 1  # 采用sigmoid模式, 则这里为单类
label_endswith = "_mask"
image_scale = (256, 256)
# **********************************-albumentations-******************************** #
_albu_augentations = [
    dict(type='Rotate', limit=360, interpolation=1, border_mode=0,
         value=None, mask_value=None, always_apply=False, p=0.5),
    dict(type='ShiftScaleRotate', shift_limit=0.2, scale_limit=0.1,
         rotate_limit=15, interpolation=1, border_mode=0, p=0.5),
    dict(type="VerticalFlip", p=0.5),
    dict(type="HorizontalFlip", p=0.5),
    dict(type='RandomResizedCrop', height=image_scale[0], width=image_scale[1], scale=(0.7, 1.0),
         ratio=(0.9, 1.2), interpolation=1, always_apply=False, p=1.0),
    dict(type='RandomBrightnessContrast', brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    dict(type='ElasticTransform', alpha=100, sigma=25, alpha_affine=30, border_mode=0, p=0.5),
]

_albu_test_transforms = [
    dict(type='Resize', height=image_scale[0], width=image_scale[1], interpolation=1),
]
# *************************************pipeline************************************ #

train_pipeline = [
    # pre_pipeline
    # dict(type='RandomRatioSelectSample', class_ratio_dict=_class_ratios_dict),  # 控制比例选择图片
    dict(type='LoadImageFromFile'),
    dict(type='LoadMaskFromFile', num_class=num_classes),  # 分割是需要导入mask图片文件
    # dict(type='CustomLoadRefOfflineImageFromFile', extensions='_normal.png'),  # 导入参考图
    # aug_pipeline
    dict(type='AlbuClsSeg', transforms=_albu_augentations),  # albumentations interface
    # dict(type='RandomNormalNoise', mode=0, prob=0.5, noise_scale=5),  # 高斯噪声
    dict(type='Resize', img_scale=image_scale, keep_ratio=True),  # multi_scale
    # dict(type='Resize', img_scale=[(572, 572)], keep_ratio=True),  # multi_scale
    dict(type='RandomShift', mode=0, prob=0.5, shift_range=(64, 64), border_value=(0, 0, 0)),  # border_value for mask
    dict(type='Normalize', **img_norm_cfg),
    # post_pipeline
    # dict(type='ProcessRefImage', is_debug=True),  # 参考图
    dict(type='EncodeMaskToOneHot', num_class=num_classes, color_values=class_order_dict),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'], meta_keys=('path', 'mask_path'))
]

val_pipeline = [
    # pre_pipeline
    dict(type='LoadImageFromFile'),
    dict(type='LoadMaskFromFile', num_class=num_classes),  # 分割是需要导入mask图片文件
    # dict(type='CustomLoadRefOfflineImageFromFile', extensions='_normal.png'),  # 导入参考图
    # dict(type='AlbuClsSeg', transforms=_albu_test_transforms),  # 验证过程不需要做数据增强
    dict(type='Normalize', **img_norm_cfg),
    # post_pipeline
    # dict(type='ProcessRefImage', is_debug=True),
    dict(type='EncodeMaskToOneHot', num_class=num_classes, color_values=class_order_dict),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'], meta_keys=('path', 'mask_path'))
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=image_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='Normalize', **img_norm_cfg),
            # post_pipeline
            # dict(type='ProcessRefImage', is_debug=True),
            dict(type='ImageToTensor', keys=['img', ]),
            dict(type='Collect', keys=['img', ]),
        ])
]

# *******************************datalayer / data*********************************** #
data = dict(
    samples_per_gpu=16,  # train用
    workers_per_gpu=12,  # train用
    eval_samples_per_gpu=1,  # val和eval用
    eval_workers_per_gpu=0,
    train=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type=dataset_type,
            img_prefix=data_root + 'train/',
            pipeline=train_pipeline,
            label_endswith=label_endswith)),
    val=dict(
        type=dataset_type,
        img_prefix=data_root + 'val/',
        pipeline=val_pipeline,
        label_endswith=label_endswith),

    test=dict(
        type=dataset_type,
        img_prefix=data_root + 'test/',
        pipeline=test_pipeline,
        label_endswith=label_endswith),
)
