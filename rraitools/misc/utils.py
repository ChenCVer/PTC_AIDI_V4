import cv2
import numpy as np
import torch


def generate_heatmap(shape, pt, sigma, use_opencv=False, truncate_thre=0.0003):
    """
    :param shape: shape of heatmap hw
    :param pt:  array, points xy
    :param sigma:  标准差 (xx,xx)
    :param flag: 0=高斯模糊模式，1=numpy模式
    :return:  高斯热图 (值为0~1)
    """

    if use_opencv:
        # when use opencv, gauss must be not in float.
        heatmap = np.zeros((int(shape[0]), int(shape[1])))
        if (int(pt[0] + 0.5) >= shape[1]) \
                or (int(pt[1] + 0.5) >= shape[0]) \
                or (pt[0] < 0) or (pt[1] < 0):
            return heatmap

        assert int(sigma[0]) % 2 != 0 and int(sigma[1]) % 2 != 0, 'sigma must be odd'
        heatmap[int(pt[1] + 0.5), int(pt[0] + 0.5)] = 1
        heatmap = cv2.GaussianBlur(heatmap, sigma, 0)
        am = np.max(heatmap)
        heatmap = heatmap / am
    else:
        # numpy format
        xx, yy = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        heatmap = np.exp(-(((xx - pt[0]) ** 2) / (2 * sigma[0] ** 2) +
                           ((yy - pt[1]) ** 2) / (2 * sigma[1] ** 2)))
        heatmap[heatmap < truncate_thre] = 0  # 过小值不要

    return heatmap


def custom_encode_onehot(label, color_dict):
    """
    Notes: mask中对应颜色替换为类别, 构建单通道mask图.
    """
    semantic_map = []

    for class_name, color_value in color_dict.items():
        class_index = int(class_name[:class_name.find(".")])
        equality = np.equal(label, color_value)
        class_map = np.all(equality, axis=-1).astype(int)
        class_map = np.expand_dims(np.where(class_map == 1, class_index, 0), axis=-1)
        semantic_map.append(class_map)

    semantic_map = np.max(np.concatenate(semantic_map, axis=2), axis=-1)

    return semantic_map


def encode_onehot(label, color_dict):
    """
        Convert a segmentation image label array to one-hot format
        by replacing each pixel value with a vector of length num_classes

        # Arguments
            label: The 2D array segmentation image label
            label_values

        # Returns
            A 2D array with the same width and hieght as the input, but
            with a depth size of num_classes
        """
    semantic_map = []
    for cls_id, color in color_dict.items():  # rgb颜色列表
        equality = np.equal(label, color)
        class_map = np.all(equality, axis=-1).astype(int)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)
    return semantic_map


def decode_onehot_from_tensor(predict, color_list, num_class, is_predict, model='rgb'):
    if num_class == 1:
        if is_predict:
            predict = torch.sigmoid(predict)
        return predict
    else:
        if is_predict:
            predict = torch.softmax(predict, dim=1)
        label = torch.argmax(predict, dim=1)
        if model == 'bgr':
            color_list = list(map(lambda x: x[::-1], list(color_list.values())))
        label = torch.tensor(color_list)[label.long()]
        return label.permute(0, 3, 1, 2).to(predict.device)


def decode_onehot(label, color_list=None):
    """
        Reverse decode one-hot label to bgr image.
        If label's shape equals to 2, meaning single class, color_list can be None.

        :param label: The 2D or 3D  array segmentation image label
        :param color_list: list or None,
        :return: numpy.ndarray: bgr image
        """
    if len(label.shape) == 2:  # 单类
        label_bgr = np.uint8([label, label, label])
        label_bgr = np.transpose(label_bgr, [1, 2, 0])
    else:
        label_bgr = np.argmax(label, axis=-1)
        label_bgr = np.array(color_list)[label_bgr.astype(int)]
    return np.uint8(label_bgr)
