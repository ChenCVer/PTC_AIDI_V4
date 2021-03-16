_base_ = [
    '../_base_/datasets/segmentation/electronic_counter.py',
    '../_base_/models/segmentation/Unet_single.py',
    '../_base_/schedules/segmentation/sgd_linear.py',
    '../_base_/misc/segmentation/default_runtime_single.py'
]