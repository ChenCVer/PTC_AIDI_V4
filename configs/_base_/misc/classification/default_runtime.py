# -*- coding:utf-8 -*-
import time

# ---------------------------------------------------------- #
# train相关设置:
# checkpoint config
checkpoint_config = dict(interval=5)  # 每隔5轮保存一次模型
# 训练流程
workflow = [('train', 5), ('val', 1)]
# resume
resume_from = None
load_from = None
# load_from = '/home/cxj/Desktop/resnet18-5c106cde.pth'
# 中间文件保存路径
work_dir = './work_dirs/'
# log config(日志输出相关)
_timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
log_config = dict(
    interval=10,
    is_disp_on_terminal=True,
    log_level='INFO',
    hooks=[dict(type='TextLoggerHook'),
           dict(type="ClsTensorboardHook", show_cam=True, interval=50),
           ]
)
# ---------------------------------------------------------- #
# eval(eval)相关设置, 给出评价指标数值和预测结果, eval过程不会计算loss,
# val_data的loss计算是在val_step过程个计算的, 评估过程只给出相关评估指标.
# mmdetection中的eval实际上是对val数据计算相关指标.
# TODO: 多分类提供acc和confusion_matrix
evaluation = dict(interval=5,
                  metric=['acc', 'confusion_matrix'],
                  use_sigmoid=False,
                  thres_score=0.5)
