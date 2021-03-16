# -*- coding:utf-8 -*-
import time

# ---------------------------------------------------------- #
# train相关设置:
# checkpoint config
checkpoint_config = dict(interval=5)  # 每隔5轮保存一次模型
# 训练流程
workflow = [('train', 1), ]
# resume
resume_from = None
load_from = None
# 中间文件保存路径
work_dir = './work_dirs/'
# log config(日志输出相关)
_timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
log_config = dict(
    interval=10,
    is_disp_on_terminal=True,
    log_level='INFO',
    hooks=[dict(type='TextLoggerHook'),
           # dict(type="KeyPointTensorboardHook", show_cam=False, interval=50),
           ]
)
# ---------------------------------------------------------- #
evaluation = dict(interval=200,
                  metric=['acc', 'confusion_matrix'],
                  use_sigmoid=False,
                  thres_score=0.2)
# ---------------------------------------------------------- #
# test测试相关配置
# 待测试的模型
checkpoint = '/home/cxj/Desktop/epoch_120.pth'
# 结果保存路径
results_path = '/home/cxj/Desktop/results'