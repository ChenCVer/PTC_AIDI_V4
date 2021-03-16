### 训练流程说明

train：针对train数据集，核心函数train_step() -> forward_train()，在此过程中需要统计一个batch内的loss，评价指标(acc, map, batch_soft_iou)等信息；

valid：针对valid数据集，核心函数val_step() -> forward_train()，此过程和训练过程一样，统计相关loss和评价指标信息，也是在一个batch内部进行。

eval：针对valid数据集，核心函数forword_test()函数，此过程对整个验证集合进行系列指标评估信息(不统计loss信息，只统计相关评估指标比如acc，map，hard_iou，FP(R)，FN(R)等信息)。