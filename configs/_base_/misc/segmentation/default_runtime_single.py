# -*- coding:utf-8 -*-
import time
# checkpoint config
checkpoint_config = dict(interval=10)  # 每隔5轮保存一次模型
# 训练流程
workflow = [('train', 10), ('val', 1)]
# resume
resume_from = None
load_from = None
# load_from = "/home/cxj/Desktop/epoch_2000.pth"
# 中间文件保存路径
work_dir = './work_dirs/'

# log config(日志输出相关)
_timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
log_config = dict(
    interval=50,
    is_disp_on_terminal=True,
    log_level='INFO',
    hooks=[dict(type='TextLoggerHook'),
           dict(type="SegTensorboardHook",
                num_class=1,
                interval=50),  # tensorboard的interval单独设置为了节省硬盘内存开支
           ]
)

# 评估: mmdetection中的eval实际上是对val数据计算相关指标.
evaluation = dict(interval=10, metric=['soft-IOU', ])