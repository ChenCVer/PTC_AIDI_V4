import torch
from torch.nn.functional import interpolate
import numpy as np
from rraitools import ImageHelper, encode_onehot


"""
对于分割网络, 采用常见的评估指标: Soft-IOU, Hard-IOU, PA
"""


class SegmentationEvaluator(object):
    """
    Provide some evaluator for semantic segmentation, mainly includes iou, pa and fn, fp
    """

    def __init__(self, exclude_channel=0, eps=1e-6):
        """

        :param eps: float: a constant
        :param exclude_channel: int: 排除计算的通道数，默认为0，通常表示背景类，
                                     如果所有通道数都计算，传入参数为None，
                                     如果有两个以上的通道数不计算，传入为list，如[1, 2]
        """
        self.eps = eps

        if exclude_channel is None or isinstance(exclude_channel, list):
            self.exclude_channel = exclude_channel
        else:
            self.exclude_channel = [exclude_channel]

    def __get_channel_list(self, channel):
        """
        Decide which channels are chosen to compute.

        :param channel: int: 总的通道数
        :return: list, 去除某些通道之后的通道列表, example=[1, 3, 4, 5]
        """
        assert channel != 0, 'channel cannot be zero!'
        channel_list = list(range(channel))

        if channel != 1 and self.exclude_channel is not None:
            for index in self.exclude_channel:
                if index < 0:
                    index = channel + index
                channel_list.remove(index)

        return channel_list

    def calc_accuracy_score(self, predictive, target):
        pass

    def calc_hard_iou_pa_numpy(self, predictive, target):
        """
        Compute hard iou and pixel accuracy of each defect of predictive and target in segmentator.
        This function computes iou based on thresholded image.

        :param predictive: numpy.ndarray, value in (0, 255). Estimated targets as returned by a segmentator.
        :param target: numpy.ndarray, value in (0, 255). Ground truth (correct) target values.

        :return: float, float: iou , pa
        """
        predictive = ImageHelper.convert_img_to_uint8(predictive)
        target = ImageHelper.convert_img_to_uint8(target)

        if len(predictive) == 2:
            predictive = np.expand_dims(predictive, axis=2)
            target = np.expand_dims(target, axis=2)

        channel = predictive.shape[2]
        channel_list = self.__get_channel_list(channel)
        predictive_flat = np.reshape(predictive, [-1, channel])
        target_flat = np.reshape(target, [-1, channel])

        iou_score_list = []
        pa_score_list = []

        for c in channel_list:
            predictive_ch = (predictive_flat[:, c] == 255)
            target_ch = (target_flat[:, c] == 255)
            _and = float(np.sum(np.bitwise_and(predictive_ch, target_ch)))
            _or = float(np.sum(np.bitwise_or(predictive_ch, target_ch)))
            _pred = float(np.sum(predictive_ch))
            iou_score_list.append((_and + self.eps) / (_or + self.eps))
            pa_score_list.append((_and + self.eps) / (_pred + self.eps))

        return sum(iou_score_list) / len(channel_list), sum(pa_score_list) / len(channel_list)

    def calc_soft_iou_pa_numpy(self, predictive, target):
        """
        Compute mean soft iou and pixel accuracy of each defect of predictive and target in segmentator.
        This function computes iou based on raw image. For the same two image,
         soft iou is usually lower than hard iou.

        :param predictive: numpy.ndarray, value in 0~255. Estimated targets as returned by a segmentator.
        :param target: numpy.ndarray, value in 0~255. Ground truth (correct) target values.

        :return: float, float: iou , pa
        """
        predictive = ImageHelper.convert_img_to_float32(predictive)
        target = ImageHelper.convert_img_to_float32(target)

        if len(predictive) == 2:
            predictive = np.expand_dims(predictive, axis=2)
            target = np.expand_dims(target, axis=2)
        channel = predictive.shape[2]

        predictive_flat = np.reshape(predictive, [-1, channel])
        target_flat = np.reshape(target, [-1, channel])

        intersection = 2.0 * np.sum(predictive_flat * target_flat, axis=0) + self.eps
        denominator = np.sum(predictive_flat, axis=0) + np.sum(target_flat, axis=0) + self.eps
        iou = intersection / denominator
        # pa表示预测正确的像素点占总像素的比例, 与recall类似
        pa = intersection / (2.0 * np.sum(target_flat, axis=0) + self.eps)

        channel_list = self.__get_channel_list(channel)

        iou = iou[channel_list].mean()
        pa = pa[channel_list].mean()

        return iou, pa

    def __calc_soft_iou_intersection(self, mask, contour, index):
        temp = np.zeros(mask.shape, dtype=np.uint8)
        ImageHelper.draw_contours(temp, contour, index=index)
        bl_p = (mask == 255)
        bl_m = (temp == 255)
        iou_intersection = np.sum(np.bitwise_and(bl_p, bl_m))

        return iou_intersection

    # label缺陷个数,预测缺陷个数，误报，漏报
    def calc_fp_fn(self, predictive, target):
        """
        Count fp number and fn number of defects of predictive and target in segmentator.
        fp (false positive) regards normal as abnormal, fn (false negative) regards abnormal as normal.
        This function supports to calculate single classifier, binary classifier and multiple classifier.

        :param predictive: numpy.ndarray, value in 0~255. Estimated targets as returned by a segmentator.
        :param target: numpy.ndarray, value in 0~255. Ground truth (correct) target values.

        :return: int, int, int, int: predictive defect number, target defect number, fp number, fn number
        """
        # FP: 误报
        # FN: 漏报
        # 兼容单类，二分类和多分类
        predictive = ImageHelper.convert_img_to_uint8(predictive)
        target = ImageHelper.convert_img_to_uint8(target)

        if len(predictive) == 2:
            predictive = np.expand_dims(predictive, axis=2)
            target = np.expand_dims(target, axis=2)

        channel = predictive.shape[2]
        channel_list = self.__get_channel_list(channel)

        num_p_list = []
        num_m_list = []
        fp_list = []
        fn_list = []
        for c in channel_list:
            cont_pred = ImageHelper.calc_contours(predictive[..., c])
            cont_mask = ImageHelper.calc_contours(target[..., c])
            num_m = len(cont_mask)
            num_p = len(cont_pred)

            mask = target[..., c]
            pred = predictive[..., c]
            hit = 0
            for cm in range(len(cont_mask)):
                iou_intersection = self.__calc_soft_iou_intersection(pred, cont_mask, index=cm)
                if iou_intersection > 0:
                    hit += 1

            hit_p = 0
            for cp in range(len(cont_pred)):
                iou_intersection = self.__calc_soft_iou_intersection(mask, cont_pred, index=cp)
                if iou_intersection > 0:
                    hit_p += 1

            fp = num_p - hit_p
            fn = num_m - hit

            num_p_list.append(num_p)
            num_m_list.append(num_m)
            fp_list.append(fp)
            fn_list.append(fn)

        return sum(num_p_list), sum(num_m_list), sum(fp_list), sum(fn_list)

    def calc_soft_iou_tensor(self, predictive, target):
        """
        Compute mean soft iou of each defect of predictive and target in segmentator.
        This function computes iou based on raw tensor.

        :param predictive: torch.tensor, value in 0~255. Estimated targets as returned by a segmentator.
        :param target: torch.tensor, value in 0~255. Ground truth (correct) target values.

        :return: torch.tensor: iou
        """

        channel = predictive.size()[1]
        channel_list = self.__get_channel_list(channel)

        predict = predictive.contiguous().view(predictive.shape[0], channel, -1)
        label = target.contiguous().view(predict.shape[0], channel, -1)

        iou_score_sum = 0

        for c in channel_list:
            num = 2 * torch.sum(predict[:, c, :] * label[:, c, :], dim=1) + self.eps
            den = torch.sum(predict[:, c, ...], dim=1) + torch.sum(label[:, c, ...], dim=1) + self.eps
            iou_score = num / den
            iou_score_sum += iou_score.mean()

        return iou_score_sum / len(channel_list)


class EncodeSegMaskToOneHot(object):
    """
    此函数可以对mask进行编码. 该函数可以定制
    """
    def __init__(self, num_class=1, color_values=None):
        self.num_class = num_class
        self.color_values = color_values

    def __call__(self, target):
        if self.num_class > 1:
            target = encode_onehot(target, self.color_values)
        else:
            if len(target.shape) == 3:
                target = target[..., 0]
            target = np.where(target > 0, 1.0, 0.0)[..., None]

        return target


def eval_seg_metrics(prediction,
                     targets,
                     metric,
                     cfg=None):
    # encode target
    class_order_dict = cfg.get("class_order_dict")  # 获取颜色
    num_classes = cfg.get("num_classes")  # 获取类别数
    encode_target = EncodeSegMaskToOneHot(num_class=num_classes, color_values=class_order_dict)

    seg_evaluator = SegmentationEvaluator()
    eval_metrics = {"mode": "eval"}  # 方便日志标记
    soft_IOU_list = []
    hard_IOU_list = []
    soft_pa_list = []
    hard_pa_list = []

    for idx, target in enumerate(targets):
        target = encode_target(target)
        if num_classes == 1:
            predict = prediction[idx][..., 0:1] / 255.0
        else:
            predict = prediction[idx] / 255.0

        if "soft-IOU" in metric:
            # 不需要用阈值进行分割
            soft_iou, soft_pa = seg_evaluator.calc_soft_iou_pa_numpy(predict, target)
            soft_IOU_list.append(soft_iou)
            soft_pa_list.append(soft_pa)
            eval_metrics["soft-iou"] = sum(soft_IOU_list) / len(soft_IOU_list)
            eval_metrics["soft-pa"] = sum(soft_pa_list) / len(soft_pa_list)

        if "hard-IOU" in metric:
            hard_iou, hard_pa = seg_evaluator.calc_hard_iou_pa_numpy(predict, target)
            hard_IOU_list.append(hard_iou)
            hard_pa_list.append(hard_pa)
            eval_metrics["hard-iou"] = sum(hard_IOU_list) / len(hard_IOU_list)
            eval_metrics["hard-pa"] = sum(hard_pa_list) / len(hard_pa_list)

    return eval_metrics