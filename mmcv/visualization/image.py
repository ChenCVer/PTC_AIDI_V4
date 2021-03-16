# Copyright (c) Open-MMLab. All rights reserved.
import cv2
import numpy as np

from mmcv.image import imread, imwrite
from .color import color_val
from PIL import Image, ImageDraw, ImageFont

FONT_SIZE = 12
LINE_WIDTH = 3
IMAGE_FONT = ImageFont.truetype(u"NotoSansCJK-Black.ttc", FONT_SIZE)

COLOR_LIST = ["red",
              "darkcyan",
              "blue",
              "orange",
              "green",
              "purple",
              "deeppink",
              "ghostwhite",
              "darkcyan",
              "olive",
              "orange",
              "orangered",
              "darkgreen"]

COLOR_CV2_LIST = [(0, 0, 255),
                  (255, 0, 0),
                  (0, 97, 255),
                  (0, 255, 0),
                  (64, 125, 255),
                  (255, 255, 255)]


def imshow(img, win_name='', wait_time=0):
    """Show an image.

    Args:
        img (str or ndarray): The image to be displayed.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
    """
    cv2.imshow(win_name, imread(img))
    if wait_time == 0:  # prevent from hangning if windows was closed
        while True:
            ret = cv2.waitKey(1)

            closed = cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1
            # if user closed window or if some key pressed
            if closed or ret != -1:
                break
    else:
        ret = cv2.waitKey(wait_time)


def imshow_bboxes(img,
                  bboxes,
                  colors='green',
                  top_k=-1,
                  thickness=1,
                  show=True,
                  win_name='',
                  wait_time=0,
                  out_file=None):
    """Draw bboxes on an image.

    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (list or ndarray): A list of ndarray of shape (k, 4).
        colors (list[str or tuple or Color]): A list of colors.
        top_k (int): Plot the first k bboxes only if set positive.
        thickness (int): Thickness of lines.
        show (bool): Whether to show the image.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
        out_file (str, optional): The filename to write the image.
    """
    img = imread(img)

    if isinstance(bboxes, np.ndarray):
        bboxes = [bboxes]
    if not isinstance(colors, list):
        colors = [colors for _ in range(len(bboxes))]
    colors = [color_val(c) for c in colors]
    assert len(bboxes) == len(colors)

    for i, _bboxes in enumerate(bboxes):
        _bboxes = _bboxes.astype(np.int32)
        if top_k <= 0:
            _top_k = _bboxes.shape[0]
        else:
            _top_k = min(top_k, _bboxes.shape[0])
        for j in range(_top_k):
            left_top = (_bboxes[j, 0], _bboxes[j, 1])
            right_bottom = (_bboxes[j, 2], _bboxes[j, 3])
            cv2.rectangle(
                img, left_top, right_bottom, colors[i], thickness=thickness)

    if show:
        imshow(img, win_name, wait_time)
    if out_file is not None:
        imwrite(img, out_file)


def imshow_det_bboxes(img,
                      bboxes,
                      labels,
                      class_names=None,
                      score_thr=0,
                      show=True,
                      win_name='',
                      wait_time=0,
                      out_file=None):
    """Draw bboxes and class labels (with scores) on an image.

    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5).
        labels (ndarray): Labels of bboxes.
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown.
        show (bool): Whether to show the image.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
        out_file (str or None): The filename to write the image.
    """
    assert bboxes.ndim == 2
    assert labels.ndim == 1
    assert bboxes.shape[0] == labels.shape[0]
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5
    img = imread(img)

    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]

    image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    # 在这里将OpenCV转换成PIL.Image格式, 然后在转换为Opencv格式, 这样做便于遇到中文场景时候显示乱码
    for bbox, label in zip(bboxes, labels):
        bbox_int = bbox.astype(np.int32)
        # 现将img转换成PIL格式的image
        # 选颜色的时候, 根据label进行选择, 这样每一个label有固定的颜色
        COLOR = COLOR_LIST[label % len(COLOR_LIST)]
        for w_i in range(LINE_WIDTH):
            x_left = bbox_int[0] + w_i
            y_top = bbox_int[1] + w_i
            x_right = bbox_int[2] - w_i
            y_down = bbox_int[3] - w_i
            draw.rectangle((x_left, y_top, x_right, y_down), None, COLOR)

        label_text = class_names[label] if class_names is not None else 'cls {}'.format(label)
        if len(bbox) > 4:
            label_text += '|{:.02f}'.format(bbox[-1])
            draw.text((bbox_int[0], bbox_int[1] - FONT_SIZE - 5), label_text, font=IMAGE_FONT, fill=COLOR)
    # 将image转换为opencv格式
    img = np.asarray(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if show:
        imshow(img, win_name, wait_time)
    if out_file is not None:
        imwrite(img, out_file)


def add_det_bboxes(img,
                   bboxes,
                   labels,
                   class_names=None,
                   score_thr=0):
    """Draw bboxes and class labels (with scores) on an image.

    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5).
        labels (ndarray): Labels of bboxes.
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown.
    """
    assert bboxes.ndim == 2
    assert labels.ndim == 1
    assert bboxes.shape[0] == labels.shape[0]
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5
    img = imread(img)

    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]

    image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    # 在这里将OpenCV转换成PIL.Image格式, 然后在转换为Opencv格式, 这样做便于遇到中文场景时候显示乱码
    for bbox, label in zip(bboxes, labels):
        bbox_int = bbox.astype(np.int32)
        # 现将img转换成PIL格式的image
        # 选颜色的时候, 根据label进行选择, 这样每一个label有固定的颜色
        COLOR = COLOR_LIST[label % len(COLOR_LIST)]
        for w_i in range(LINE_WIDTH):
            x_left = bbox_int[0] + w_i
            y_top = bbox_int[1] + w_i
            x_right = bbox_int[2] - w_i
            y_down = bbox_int[3] - w_i
            draw.rectangle((x_left, y_top, x_right, y_down), None, COLOR)

        label_text = class_names[label] if class_names is not None else 'cls {}'.format(label)
        if len(bbox) > 4:
            label_text += '|{:.02f}'.format(bbox[-1])
            draw.text((bbox_int[0], bbox_int[1] - FONT_SIZE - 5), label_text, font=IMAGE_FONT, fill=COLOR)
    # 将image转换为opencv格式
    img = np.asarray(image)
    return img


def add_gt_bboxes(img, gt_bboxes, gt_labels, class_names=None):
    img = imread(img)
    image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    # 在这里将OpenCV转换成PIL.Image格式, 然后在转换为Opencv格式, 这样做便于遇到中文场景时候显示乱码
    # 将gt画上去
    for gt_bbox, gt_label in zip(gt_bboxes, gt_labels):
        gt_bbox_int = gt_bbox.astype(np.int32)
        COLOR = COLOR_LIST[(gt_label - 1) % len(COLOR_LIST)]
        for w_i in range(LINE_WIDTH):
            x_left = gt_bbox_int[0] + w_i
            y_top = gt_bbox_int[1] + w_i
            x_right = gt_bbox_int[2] - w_i
            y_down = gt_bbox_int[3] - w_i
            draw.rectangle((x_left, y_top, x_right, y_down), None, COLOR)

        # 如果数据集的类别是从0开始, 则这里是class_names[gt_label], 如果是从1开始, 为: class_names[gt_label - 1]
        gt_label_text = class_names[gt_label] if class_names is not None else 'cls {}'.format(gt_label)
        gt_label_text += '|(GT)'
        draw.text((gt_bbox_int[0], gt_bbox_int[1] - FONT_SIZE - 5), gt_label_text, font=IMAGE_FONT, fill=COLOR)
    # 将image转换为opencv格式
    img = np.asarray(image)

    return img
