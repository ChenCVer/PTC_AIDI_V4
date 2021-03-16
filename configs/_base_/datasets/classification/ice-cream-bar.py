"""
该配置主要是实现, 如果文件夹分布形式是:
0.bg - [class_1_folder,..., class_n_folder],
1.ng - [class_1_folder,..., class_n_folder]
本配置可以实现抽取0.bg和1.ng任意个类别进行分类训练.
"""
dataset_type = 'ClsDataset'
data_root = '/home/cxj/Desktop/data/ice_cream_bar_cls_128/'

img_norm_cfg = dict(
    mean_bgr=[0., 0., 0.],
    std_bgr=[1., 1., 1.], to_rgb=True)
# ********************************************************************************** #
# _class_list = ["0.pullution", "1.mineral_line", "2.blace_spot", "3.light_spot", "4.ok"]
_class_list = None
_class_ratios_dict = {"0.pullution": [2, ],
                      "1.mineral_line": [1, ],
                      "2.blace_spot": [1, ],
                      "3.light_spot": [3, ],
                      "4.ok": [2, ],
                      }  # 此_class_ratios_dict必须有
img_scale = (128, 128)  # 输入图像尺度
# **********************************-albumentations-******************************** #
# 为了兼容album数据增强库操作, album的标准写法
_albu_train_transforms = [
    dict(type='Rotate', limit=15, interpolation=1, border_mode=0,
         value=None, mask_value=None, always_apply=False, p=0.5),
    dict(type='ShiftScaleRotate', shift_limit=0.2, scale_limit=0.2,
         rotate_limit=5, interpolation=1, border_mode=0, p=0.5),
    dict(type='RandomResizedCrop', height=img_scale[0], width=img_scale[1], scale=(0.7, 1.0),
         ratio=(0.75, 1.33333333), always_apply=False, p=0.7),
    dict(type='RandomBrightnessContrast', contrast_limit=0.2, brightness_limit=0.2, p=0.5),
    dict(type='HorizontalFlip', p=0.5),
    dict(type='VerticalFlip', p=0.5),
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
    dict(type='RandomShift', mode=1, prob=0.5, shift_range=(10, 10)),
    # 自己添加的数据增强
    dict(type='RandomCrop', mode=1, prob=0.5, pix_range_hw=(10, 10)),
    dict(type='RandomNormalNoise', mode=1, prob=0.5, noise_scale=10),
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
    # dict(type='Resize', keep_ratio=False),
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
        img_scale=img_scale,
        flip=False,
        transforms=[
            # dict(type='Resize', keep_ratio=False),
            dict(type='Normalize', **img_norm_cfg),
            # dict(type='ProcessRefImage', is_debug=True),
            dict(type='ImageToTensor', keys=['img', ]),
            dict(type='Collect', keys=['img', ]),
        ])
]

data = dict(
    samples_per_gpu=64,  # train -> batchsize
    workers_per_gpu=24,  # train -> worker
    eval_samples_per_gpu=1,  # val和eval用, eval和val的batch设定为1, 主要考虑后期每张图片大小不一致.
    eval_workers_per_gpu=0,  # val
    train=dict(
            type=dataset_type,
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
