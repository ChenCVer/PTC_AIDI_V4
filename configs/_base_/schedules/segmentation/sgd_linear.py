# optimizer
optimizer = dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))  # 防止梯度爆炸
optimizer_config = dict(grad_clip=None)  # 防止梯度爆炸

# learning policy (scheduler) config
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[900, 1400])  # 因为step=[80, 90]与max_epoch有关, 因此写在这里

total_epochs = 2000
