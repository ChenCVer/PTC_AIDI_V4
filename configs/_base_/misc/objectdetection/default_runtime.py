# yapf:enable
load_from = None
resume_from = None
workflow = [('train', 5), ("val", 1)]
work_dir = "./work_dirs"

# yapf:disable
log_config = dict(
    interval=100,
    is_disp_on_terminal=True,
    log_level='INFO',
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='DetTensorboardHook'),
    ])

checkpoint_config = dict(interval=5)
# 评估: 在验证集上进行相应的指标评估.
evaluation = dict(interval=5, metric='mAP')
