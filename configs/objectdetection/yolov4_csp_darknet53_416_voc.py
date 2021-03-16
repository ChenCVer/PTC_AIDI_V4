_base_ = [
    '../_base_/models/objectdetection/yolov4_csp_d53_416_iou.py',
    '../_base_/datasets/objectdetection/ssdd_best_cfg.py',
    '../_base_/schedules/objectdetection/schedule_273e.py',
    '../_base_/misc/objectdetection/default_runtime.py'
]