"""
RegNet: 好网络的宽度和深度是可以用量化的线性函数来解释的.
核心思想是提出了对网络设计空间进行整体估计(population estimation,意思就是所有的深度宽度之类
的最佳设计空间关系给它估计出来). 非常直观地, 如果我们能得到深度(depth),宽度(width)等等一系列
网络设计要素关于网络设计目标的函数关系(u_j=w_0+w_a*j),那么我们就很轻松地知道大概多深的网络,
多宽的网络是最佳选择.
"""
# model settings
model = dict(
    type='BaseClassifier',
    pretrained='open-mmlab://regnetx_1.6gf',
    backbone=dict(
        type='RegNet',
        in_channels=3,
        arch='regnetx_1.6gf',
        out_indices=(3,),
        style='pytorch',
        norm_eval=False,
        zero_init_residual=False,
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=5,
        # arch:800mf->672; 1.6gf->912; 3.2gf->1008; 4.0gf->1360; 6.4gf->1624;8.0gf->1920; 12gf->2240;
        in_channels=912,
        loss=dict(
            type='GeneralLosser',
            losses=[dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0), ],
        ),
    )
)

train_cfg = None
valid_cfg = None
test_cfg = dict(score_thr=0.8, use_sigmoid=False, out_cam=True)