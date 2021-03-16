import collections

dataset_type = 'SegDataset'
_data_root = '/home/cxj/Desktop/data/person-face-dataset-binary-backup/'
img_norm_cfg = dict(mean_bgr=[0.0, 0.0, 0.0],
                    std_bgr=[255.0, 255.0, 255.0], to_rgb=True)
# ********************************************************************************** #
# 离线训练不需要提供颜色列表信息
_class_ratios_dict = {"0.bg": [0],
                      "1.ng": [1]}
class_order_dict = collections.OrderedDict({
               "0.bg": (0, 0, 0),
               "1.ng": (255, 255, 255)})
num_classes = len(class_order_dict) - 1  # 采用sigmoid模式, 则这里为单类
label_endswith = "_mask"
# **********************************-albumentations-******************************** #
_albu_augentations = [
    dict(type='Resize', height=400, width=300, interpolation=1),
    dict(type='Rotate', limit=30, interpolation=1, border_mode=0,
         value=None, mask_value=None, always_apply=False, p=0.5),
    dict(type='ShiftScaleRotate', shift_limit=0.1, scale_limit=0.5,
         rotate_limit=0, interpolation=1, border_mode=0, p=0.5),
    dict(type='RandomResizedCrop', height=400, width=300, scale=(0.8, 1.0),
         ratio=(0.75, 1.33333333), interpolation=1, always_apply=False, p=1.0),
    dict(type='RandomBrightnessContrast', brightness_limit=0.5, contrast_limit=0.2, p=0.5),
]

_albu_test_transforms = [
    dict(type='Resize', height=400, width=300, interpolation=1),
]
# *************************************pipeline************************************ #

train_pipeline = [
    # pre_pipeline
    # dict(type='RandomRatioSelectSample', class_ratio_dict=_class_ratios_dict),  # 控制比例选择图片
    dict(type='LoadImageFromFile'),
    dict(type='LoadMaskFromFile', num_class=num_classes),  # 分割是需要导入mask图片文件
    # dict(type='CustomLoadRefOfflineImageFromFile', extensions='_normal.png'),  # 导入参考图
    # aug_pipeline
    dict(type='AlbuClsSeg', transforms=_albu_augentations),
    dict(type='RandomNormalNoise', mode=1, prob=0.5, noise_scale=10),
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
    dict(type='AlbuClsSeg', transforms=_albu_test_transforms),  # 验证过程不需要做数据增强
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
        img_scale=(3072, 3072),
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
    samples_per_gpu=32,       # train用
    workers_per_gpu=16,        # train用
    eval_samples_per_gpu=1,   # val和eval用
    eval_workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        img_prefix=_data_root + 'train/',
        pipeline=train_pipeline,
        img_endswith=label_endswith),
    val=dict(
        type=dataset_type,
        img_prefix=_data_root + 'val/',
        pipeline=val_pipeline,
        img_endswith=label_endswith),

    test=dict(
        type=dataset_type,
        img_prefix=_data_root + 'test/',
        pipeline=test_pipeline,
        img_endswith=label_endswith),
)