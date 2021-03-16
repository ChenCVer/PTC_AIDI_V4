_base_ = [
    '../_base_/datasets/segmentation/offline_magnetic_multi.py',
    '../_base_/models/segmentation/Unet_multi.py',
    '../_base_/schedules/segmentation/sgd_linear.py',
    '../_base_/misc/segmentation/default_runtime_multi.py'
]