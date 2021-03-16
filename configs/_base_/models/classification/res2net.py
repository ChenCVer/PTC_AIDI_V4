# model settings
"""
改进: Res2Net在ResNet的核心改进是,修改ResNet中bottleneck标准的1-3-1结构, 取而代之为“4scale-(3x3)”
     残差分层架构,在一个给定的残差块中使用分层的、层叠的特征组(称为“scale”),取代了通用的单个3x3卷积核.

小技巧: Res2Net喜欢高级数据增强, 比如mix-up、CutMix等. 你可以看到使用这些工具时验证损失会急剧下降,
       因此强烈建议使用Res2Net进行大量的数据增强.
"""
model = dict(
    type='BaseClassifier',
    pretrained='open-mmlab://res2net101_v1d_26w_4s',
    backbone=dict(
        type='Res2Net',
        in_channels=3,
        depth=101,  # 50, 101, 152
        scales=4,   # Res2Net degenerates to ResNet when scales = 1.
        base_width=26,
        out_indices=(3,),
        style='pytorch',
        norm_eval=False,
        zero_init_residual=False,
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=5,
        in_channels=2048,  # depth:50/101/2048->in_channel2048;
        loss=dict(
            type='GeneralLosser',
            losses=[dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0), ],
        ),
    )
)

train_cfg = None
valid_cfg = None
test_cfg = dict(score_thr=0.8, use_sigmoid=False, out_cam=True)