# optimizer
# 一般来说, 权重衰减会应用到网络中所有需要学习的参数上面, 然而如果仅仅将权重衰减应用到conv层和fc层的w参数,
# 而对其biases参数和BN层的gamma和beta参数不进行衰减, 效果会更好.
optimizer = dict(type='SGD',
                 lr=0.001,
                 momentum=0.9,
                 weight_decay=0.0005,
                 paramwise_cfg=dict(norm_decay_mult=0.,
                                    bias_decay_mult=0.,
                                    bypass_duplicate=True))
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2000,  # same as burn-in in darknet
    warmup_ratio=0.1,
    step=[218, 246])
# runtime settings
total_epochs = 273