"""
该配置主要是实现, 如果文件夹分布形式是: 
0.bg - [class_1_folder,..., class_n_folder],
1.ng - [class_1_folder,..., class_n_folder]
本配置可以实现抽取0.bg和1.ng任意个类别进行分类训练.
"""
dataset_type = 'ClsDataset'
data_root = '/home/cxj/Desktop/data/cat-and-dog-backup/'
img_norm_cfg = dict(
    mean_bgr=[0.0, 0.0, 0.0],
    std_bgr=[1.0, 1.0, 1.0], to_rgb=False)
# ********************************************************************************** #
# _class_list = ["block", "car", "character", "graph"]
_class_list = None
_class_ratios_dict = {"0.bg": [1, ],
                      "1.ng": [1, ]}  # 此_class_ratios_dict必须有
# **********************************-albumentations-******************************** #
# 为了兼容album数据增强库操作, album的标准写法
_albu_train_transforms = [
    dict(type='Resize', height=280, width=280, interpolation=1),
    dict(type='Rotate', limit=10, interpolation=1, border_mode=0,
         value=None, mask_value=None, always_apply=False, p=0.5),
    dict(type='ShiftScaleRotate', shift_limit=0.1, scale_limit=0.5,
         rotate_limit=0, interpolation=1, border_mode=0, p=0.5),
    dict(type='RandomResizedCrop', height=256, width=256, scale=(0.7, 1.0),
         ratio=(0.75, 1.33333333), interpolation=1, always_apply=False, p=1.0),
    dict(type='RandomBrightnessContrast', brightness_limit=0.5, contrast_limit=0.2, p=0.5),
]

_albu_test_transforms = [
    dict(type='Resize', height=256, width=256, interpolation=1),
]
# *************************************pipeline************************************ #
train_pipeline = [
    # pre_pipeline
    dict(type='RandomRatioSelectSample',
         class_list=_class_list,
         class_ratio_dict=_class_ratios_dict),
    dict(type='LoadImageFromFile'),
    # dict(type='LoadRefImageFromFile', root=_data_root, ref_path=os.path.join(_data_root, 'ref')),  # ref 操作
    # aug_pipeline
    dict(type='AlbuClsSeg', transforms=_albu_train_transforms),  # 调用album官方实现的数据增强
    dict(type='RandomNormalNoise', mode=1, prob=0.5, noise_scale=5),  # 自己添加的数据增强
    dict(type='Normalize', **img_norm_cfg),
    # post_pipeline
    # dict(type='ProcessRefImage', is_debug=True),  # ref操作
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'label'])
]

val_pipeline = [
    # pre_pipeline
    dict(type='LoadImageFromFile'),
    # dict(type='CustomLoadRefOfflineImageFromFile', extensions='_normal.png'),  # 导入参考图
    dict(type='AlbuClsSeg', transforms=_albu_test_transforms),  # 验证过程不需要做数据增强
    dict(type='Normalize', **img_norm_cfg),
    # post_pipeline
    # dict(type='ProcessRefImage', is_debug=True),  # ref操作
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'label'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(256, 256),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='Normalize', **img_norm_cfg),
            # dict(type='ProcessRefImage', is_debug=True),
            dict(type='ImageToTensor', keys=['img', ]),
            dict(type='Collect', keys=['img', ]),
        ])
]

data = dict(
    samples_per_gpu=64,  # train -> batchsize
    workers_per_gpu=16,  # train -> worker
    eval_samples_per_gpu=1,  # val和eval用, eval和val的batch设定为1, 主要考虑后期每张图片大小不一致.
    eval_workers_per_gpu=0,  # val
    train=dict(type=dataset_type,
               img_prefix=data_root + 'train/',
               pipeline=train_pipeline,
               gather_flag=True),
    val=dict(type=dataset_type,
             img_prefix=data_root + 'val/',
             pipeline=val_pipeline),
    test=dict(type=dataset_type,
              img_prefix=data_root + 'test/',
              pipeline=test_pipeline),
)