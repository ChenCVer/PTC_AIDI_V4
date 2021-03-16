# -*- coding:utf-8 -*-
import time
import collections as collect

class_order_dict_ = collect.OrderedDict({
               "0.bg": (0, 0, 0),
               "1.ng": (255, 255, 255)})

# checkpoint config
checkpoint_config = dict(interval=5)  # 每隔5轮保存一次模型
# 训练流程
workflow = [('train', 10), ('val', 1)]
# resume
resume_from = None
load_from = None
# 中间文件保存路径
work_dir = './work_dirs/'

# log config(日志输出相关)
_timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
log_config = dict(
    interval=100,
    is_disp_on_terminal=True,
    log_level='INFO',
    hooks=[dict(type='TextLoggerHook'),
           dict(type="SegTensorboardHook",
                num_class=2,
                color_list=class_order_dict_,
                interval=100)]  # tensorboard的interval单独设置为了节省硬盘内存开支
)
# 评估: mmdetection中的eval实际上是对val数据计算相关指标.
evaluation = dict(interval=1,
                  metric=['soft-IOU', 'hard-IOU', 'soft-pa', 'hard-pa'],
                  thres_score=0.7)

# test测试相关配置
# 待测试的模型
checkpoint = '/home/cxj/Desktop/seg_person_face_multi.pth'
# 结果保存路径
results_path = '/home/cxj/Desktop/results'
# 分类阈值
thres_score = 0.8

