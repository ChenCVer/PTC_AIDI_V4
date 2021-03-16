import torch
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, \
                            confusion_matrix, accuracy_score

"""
分类网络的常用评价指标: https://blog.csdn.net/anshuai_aw1/article/details/88079269
Matrix(混淆矩阵): Confusion
准确度(accuracy): (TP + TN) / (TP + FP + FN + TN), 该指标一般用于平衡数据集(正负样本比例小于3).
召回率(recall, TPR): TP / (TP + FN), 该指标适用于不平衡数据集(正负样本比例大于10).
精准率(precision): TP / (TP + FP), 该指标适用于不平衡数据集, 该指标适用于衡量模型是否存在大量FP的情况
F1 - Score: 2 * (precision * recall) / (precision + recall), 同时考察precision和recall的重要性
ROC - AUC: 通过分配不同的阈值来创建不同的数据点来生成ROC曲线.ROC曲线下的面积称为AUC, AUC越高,
你的模型就越好.ROC曲线离中线越远, 模型就越好.
PR: 当数据主要位于负标签的情况下, ROC - AUC不能给出一个具有代表性的结果, 因为我们只关注TPR, 此时我们应该
采用PR曲线来衡量.
"""


def eval_cls_metrics(results,
                     targets,
                     metric=None,
                     thres_score=None):

    results_cat = torch.cat(results, dim=0)
    softmax_results = F.softmax(results_cat, dim=1)
    predict = torch.argmax(softmax_results, dim=1)
    max_inex_p = torch.max(softmax_results, dim=1)[0]
    # 这里考虑到, 虽然其类别都预测正确, 但是其概率值并不高, 这种情况,也是网络没有训练好, 泛化能力欠佳.
    # 这里,假设predict是预测正确, 但是他的概率值没达到预设值, 则变成相反类别.
    if thres_score is not None:
        predict = torch.where(max_inex_p > thres_score, predict, max(targets) - predict).tolist()
    # 以上过程是获取label和predict的过程, 以下过程计算统计指标
    eval_metrics = {"mode": "eval"}
    if 'acc' in metric:
        eval_metrics["accuracy"] = accuracy_score(targets, predict)   # accuracy

    if 'confusion_matrix' in metric:
        c_matrix = confusion_matrix(targets, predict).ravel()
        eval_metrics['confusion_matrix'] = str(c_matrix)              # confusion-matrix

    # TODO: 多分类时, 暂时不提供precison, recall和F1-score
    if max(targets) <= 1:
        if 'precision' in metric:
            eval_metrics["precison"] = precision_score(targets, predict)  # precision
        if 'recall' in metric:
            eval_metrics["recall"] = recall_score(targets, predict)       # recall
        if 'f1-score' in metric:
            eval_metrics["f1-score"] = f1_score(targets, predict)         # f1-score

    return eval_metrics