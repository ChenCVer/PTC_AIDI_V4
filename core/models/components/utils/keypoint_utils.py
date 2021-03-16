# -*- coding:utf-8 -*-
import os
import cv2
import time
import torch
import numpy as np
import torch.nn as nn
from rraitools import VisualHelper
from rraitools import FileHelper


def check_keypoint_if_in_bbox(bbox, keypoints):
    """
    :param bbox : [xmin, ymin, xmax, ymax]:
    :param keypoints: [[x1,y1],[x2,y2]....]:
    :return:
    """
    xmin, ymin, xmax, ymax = bbox
    for kpt in keypoints:
        kpt_x, kpt_y = kpt
        kpt_x, kpt_y = int(kpt_x), int(kpt_y)
        if kpt_x <= xmin or kpt_x >= xmax or kpt_y <= ymin or kpt_y >= ymax:
            return False
    return True


def display_img_and_save(img, src, img_name, tar_pts_loc_nx2, pred_pts_loc_nx2,
                         pred_prob=None, wait_time_ms=0, is_save=False):
    """
    :param img: array, single image w*h*3
    :param src: str, save path
    :param img_name: str, image name
    :param tar_pts_loc_nx2: array
    :param pred_pts_loc_nx2: array
    :param pred_prob: array, w*h
    :param wait_time_ms: int, if wait_time_ms == 0, waite key '0', else waite time wait_time_ms
    :param is_save: save or not
    :return:
    """

    img = img.squeeze()
    pred_prob = pred_prob.sum(0)
    if tar_pts_loc_nx2 is not None:
        img = VisualHelper.show_keypoint(img, keypoints_nx2=tar_pts_loc_nx2, color=(0, 255, 0),
                                         radius=5, is_show=False)
    VisualHelper.show_keypoint(img, keypoints_nx2=pred_pts_loc_nx2, color=(0, 0, 255),
                               radius=5, wait_time_ms=wait_time_ms)

    if is_save:
        time_str = time.strftime('%Y%m%d')
        tar_path = os.path.join(src, '..', 'img_predict_{}'.format(time_str))
        FileHelper.make_folders(src, tar_path)
        path = img_name.split('/')[-3:]
        path = '/'.join(path)
        tar_img_path = os.path.join(tar_path, path)
        cv2.imwrite(tar_img_path, pred_prob / pred_prob.max() * 255)


def crop_pts(pts, pt_info):
    """
    Crop pts according the information
    :param pts: array_nx2
    :param pt_info: list
    :return: array_nx2
    """
    pts_crop = np.zeros(pts.shape)
    pts_crop[:, 0] = pts[:, 0] - pt_info[1]
    pts_crop[:, 1] = pts[:, 1] - pt_info[0]
    return pts_crop


def recover_pts(pts_nx2, pt_info=None, ratio=1):
    """
    Recover points
    :param pts: array_nx2
    :param pt_info: crop information
    :param ratio: scale ratio
    :return:
    """
    if isinstance(pts_nx2, list):
        recovered_list = []
        for i, pts in enumerate(pts_nx2):
            for b in range(pts.shape[0]):
                if pt_info is None:
                    recovered_list.extend(pts[b] * ratio)
                else:
                    recovered_list.extend(crop_pts(pts[b] * ratio, pt_info[i] * -1))
        return recovered_list
    else:
        if pt_info is None:
            return pts_nx2 * ratio
        else:
            return crop_pts(pts_nx2 * ratio, pt_info * -1)


def find_pts_loc(pred_bcwh, use_subpixel=False,
                 subpixel_method=None, sigma=1,
                 use_gaussianBlur=False, use_gpu=True):
    """
    Find point from predicted probability
    :param sigma:
    :param pred_bcwh: array, batch*kpt_num*w*h
    :return: array_batch*kpt_num*2_xy, location of key points
             array_batch*kpt_num*1_prob, predicted probability of key points
    """
    if not isinstance(pred_bcwh, np.ndarray):
        if use_gpu:
            array_bcwh = pred_bcwh.cpu().detach().numpy()
        else:
            array_bcwh = pred_bcwh.detach().numpy()
    else:
        array_bcwh = pred_bcwh
    h, w = array_bcwh.shape[2:4]
    sub_array_bcwh = None
    if use_subpixel:  # 是否精确到亚像素
        sub_array_bcwh = array_bcwh.copy()
    array_bcwh = np.reshape(array_bcwh, [array_bcwh.shape[0], array_bcwh.shape[1], -1])
    index = np.argmax(array_bcwh, axis=2)  # 找到每张图中, 预测值的最大位置.
    y, x = np.unravel_index(index, (h, w))  # 将index对应到(h, w)的矩阵位置.
    pts_loc_bxcx2 = np.stack([x, y], axis=2).astype(np.float32)
    pts_value_bxcx1 = np.max(array_bcwh, axis=2)[..., None]  # 取出最大位置对应的最大值.
    if use_subpixel:
        # TODO: make speedly
        for i in range(array_bcwh.shape[0]):
            for j in range(array_bcwh.shape[1]):
                center_xy = pts_loc_bxcx2[i, j]
                if use_gaussianBlur:
                    dr = cv2.GaussianBlur(sub_array_bcwh[i, j], (sigma ** 2, sigma ** 2), 0)
                else:
                    dr = sub_array_bcwh[i, j]
                x, y = subpixel_method(dr, center_xy, sigma)
                pts_loc_bxcx2[i, j] = np.array([x, y])
    return pts_loc_bxcx2, pts_value_bxcx1


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2
    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep  # 找出相同位置的值，相当于nms操作


def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _topk(scores, K):
    batch, cat, height, width = scores.size()
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)  # batch,cat,k
    topk_ys = (topk_inds / width).int().float()  # 还原到对应的图上xy坐标上
    topk_xs = (topk_inds % width).int().float()
    return topk_scores / 255.0, topk_ys, topk_xs


def find_coor_all(spred_bcwh, topk, filter_thrsehold=0.5, use_subpixel=False,
                  subpixel_method=None, sigma=1,use_gaussianBlur=False,
                  use_gpu=True):
    assert isinstance(spred_bcwh, torch.Tensor), 'input must bu torch.tensor'
    spred_bcwh = spred_bcwh.float()
    pred_bcwh = _nms(spred_bcwh)
    topk_scores, topk_ys, topk_xs = _topk(pred_bcwh, topk)
    result = torch.stack([topk_xs, topk_ys, topk_scores], dim=3)  # batch,4,topk,3
    if use_gpu:
        result = result.cpu().numpy()
        spred_bcwh = spred_bcwh.cpu().numpy()
    else:
        result = result.numpy()
        spred_bcwh = spred_bcwh.numpy()

    pts_loc_bxcxkx2 = result[..., 0:2]
    pts_bxcxkx1 = result[..., 2:]
    # 过滤掉低于阈值的预测结果
    b, c, k = pts_loc_bxcxkx2.shape[:-1]
    pts_loc_bxcxkx2 = np.reshape(pts_loc_bxcxkx2, (-1, 2))
    pts_bxcxkx1 = np.reshape(pts_bxcxkx1, (-1, 1))
    index = pts_bxcxkx1 > filter_thrsehold
    pts_loc_bxcxkx2 = pts_loc_bxcxkx2[index[:, 0]]
    pts_bxcxkx1 = pts_bxcxkx1[index]
    pts_loc_bxcxkx2 = np.reshape(pts_loc_bxcxkx2, (b, c, -1, 2))
    pts_bxcxkx1 = np.reshape(pts_bxcxkx1, (b, c, -1, 1))

    if use_subpixel:
        # TODO 加速
        for i in range(spred_bcwh.shape[0]):
            for k in range(spred_bcwh.shape[1]):
                for j in range(pts_loc_bxcxkx2.shape[2]):
                    center_xy = pts_loc_bxcxkx2[i, k, j]
                    if use_gaussianBlur:
                        dr = cv2.GaussianBlur(spred_bcwh[i, k], (sigma ** 2, sigma ** 2), 0)
                    else:
                        dr = spred_bcwh[i, k]
                    x, y = subpixel_method(dr, center_xy, sigma)
                    pts_loc_bxcxkx2[i, k, j] = np.array([x, y])
    return pts_loc_bxcxkx2, pts_bxcxkx1


def post_process(pts, down_ratio, parameter_dict):
    """
    Recover points from aitools.augtools.LetterResize
    :param pts: array
    :param down_ratio: float, scale before use LetterResize
    :param parameter_dict: parameter dict calculated by LetterResize
    :return: recovered points
    """
    pts = pts * down_ratio
    top = parameter_dict['pad_tblr'][1]
    left = parameter_dict['pad_tblr'][0]
    scale = parameter_dict['scale']
    pts[..., 0] = pts[..., 0] - left
    pts[..., 1] = pts[..., 1] - top
    pts = pts / scale
    return pts


# 亚像素后处理
def calc_subpixel_gaussianCenter_parabola(gaussianMap2d, center_xy_int, sigma_px=1):
    """
    :param gaussianMap2d: nxm高斯热图
    :param center_xy_int: 峰值点坐标
    :param sigma_px: 和训练一样的sigma参数,默认sigma xy方向相同
    :return: 中心点x,y坐标
    """
    assert isinstance(gaussianMap2d, np.ndarray)
    assert gaussianMap2d.ndim == 2
    # 如果越界，则不进行refine操作
    center_xy_int = list(map(int, center_xy_int))
    z1 = gaussianMap2d[center_xy_int[1], center_xy_int[0]]
    if center_xy_int[1] - 1 <= 0 or center_xy_int[1] + 1 >= gaussianMap2d.shape[0]:
        offset_y = 0
    else:
        z0_y = gaussianMap2d[center_xy_int[1] - 1, center_xy_int[0]]
        z2_y = gaussianMap2d[center_xy_int[1] + 1, center_xy_int[0]]
        offset_y = 0.5 * (z0_y - z2_y) / (z0_y + z2_y - 2 * z1)
    cy = center_xy_int[1] + offset_y
    if center_xy_int[0] - 1 <= 0 or center_xy_int[0] + 1 >= gaussianMap2d.shape[1]:
        offset_x = 0
    else:
        z0_x = gaussianMap2d[center_xy_int[1], center_xy_int[0] - 1]
        z2_x = gaussianMap2d[center_xy_int[1], center_xy_int[0] + 1]
        offset_x = 0.5 * (z0_x - z2_x) / (z0_x + z2_x - 2 * z1)
    cx = center_xy_int[0] + offset_x
    return round(cx, 5), round(cy, 5)


def calc_subpixel_gaussianCenter_paraboloid(gaussianMap2d, center_xy_int, sigma_px=1):
    """
    :param gaussianMap2d: nxm高斯热图
    :param center_xy_int: 峰值点坐标
    :param sigma_px: 和训练一样的sigma参数,默认sigma xy方向相同
    :return: 中心点x,y坐标
    """
    assert isinstance(gaussianMap2d, np.ndarray)
    assert gaussianMap2d.ndim == 2
    # 查找8邻域
    h, w = gaussianMap2d.shape
    center_xy_int = list(map(int, center_xy_int))
    xx, yy = np.meshgrid(np.arange(max(center_xy_int[0] - 1, 0),
                                   min(center_xy_int[0] + 1 + 1, w)),
                         np.arange(max(center_xy_int[1] - 1, 0),
                                   min(center_xy_int[1] + 1 + 1, h)))
    if tuple(xx.shape) != (3, 3):
        # 数据不够，直接返回
        return center_xy_int
    neighbors = gaussianMap2d[yy, xx]
    a = 1 / 8.0 * (2 * (neighbors[1, 2] + neighbors[1, 0] - 2 * neighbors[1, 1])
                   + (neighbors[0, 2] + neighbors[0, 0] - 2 * neighbors[0, 1])
                   + (neighbors[2, 2] + neighbors[2, 0] - 2 * neighbors[2, 1])
                   )
    b = 1 / 8.0 * (2 * (neighbors[0, 1] + neighbors[2, 1] - 2 * neighbors[1, 1])
                   + (neighbors[0, 0] + neighbors[2, 0] - 2 * neighbors[1, 0])
                   + (neighbors[0, 2] + neighbors[2, 2] - 2 * neighbors[1, 2])
                   )
    c = 1 / 4.0 * (neighbors[0, 0] + neighbors[2, 2] - neighbors[0, 2] - neighbors[2, 0])
    d = 1 / 8.0 * ((neighbors[0, 2] - neighbors[0, 0]) + (neighbors[2, 2] - neighbors[2, 0]) + 2 * (
            neighbors[1, 2] - neighbors[1, 0]))
    e = 1 / 8.0 * ((neighbors[2, 0] - neighbors[0, 0]) + (neighbors[2, 2] - neighbors[0, 2]) + 2 * (
            neighbors[2, 1] - neighbors[0, 1]))
    cx = center_xy_int[0] + (2 * b * d - c * e) / (c * c - 4 * a * b)
    cy = center_xy_int[1] + (2 * a * e - c * d) / (c * c - 4 * a * b)
    return round(cx, 5), round(cy, 5)


def calc_subpixel_gaussianCenter_empiric(gaussianMap2d, center_xy_int, sigma_px=1):
    """
        :param gaussianMap2d: nxm高斯热图
        :param center_xy_int: 峰值点坐标
        :param sigma_px: 和训练一样的sigma参数,默认sigma xy方向相同
        :return: 中心点x,y坐标
        """
    assert isinstance(gaussianMap2d, np.ndarray)
    assert gaussianMap2d.ndim == 2
    center_xy_int = list(map(int, center_xy_int))
    gaussianMap2d_copy = gaussianMap2d.copy()
    # 求第二响应大的点位置
    gaussianMap2d_copy[center_xy_int[1], center_xy_int[0]] = 0
    index = np.argmax(gaussianMap2d_copy)
    y, x = np.unravel_index(index, gaussianMap2d.shape[0:2])
    second_coord = np.array([x, y], np.float32)
    center_coord = np.array(center_xy_int, np.float32)
    offset = (second_coord - center_coord) / np.linalg.norm((second_coord - center_coord), ord=2, axis=0)
    p = center_coord + 0.25 * offset
    return round(p[0], 5), round(p[1], 5)


def calc_subpixel_gaussianCenter_matrix(gaussianMap2d, center_xy_int, sigma_px=1):
    """
    Calculate gaussian map center in subpixel accuracy.
    reference blog: https://blog.csdn.net/houjixin/article/details/8490653/

    Args:
        gaussianMap2d: 2d gaussian map
        center_xy_int: initial center in int accuracy
        sigma_px: determine the valid neighbor range by 3*sigma principle

    Returns:
        (cx, cy): subpixel center coordinates, tuple of (cx, cy)
    """
    assert isinstance(gaussianMap2d, np.ndarray)
    assert gaussianMap2d.ndim == 2
    # extract gaussian map by initial center and 3*sigma
    h, w = gaussianMap2d.shape
    center_xy_int = list(map(int, center_xy_int))
    _3sigma = int(sigma_px * 3 + 0.5)
    xx, yy = np.meshgrid(np.arange(max(center_xy_int[0] - _3sigma, 0),
                                   min(center_xy_int[0] + _3sigma + 1, w)),
                         np.arange(max(center_xy_int[1] - _3sigma, 0),
                                   min(center_xy_int[1] + _3sigma + 1, h)))
    neighbors = gaussianMap2d[yy, xx]
    # construct equation
    # 预测值可能出现负数
    neighbors[neighbors < 0] = 0
    A_nx1 = (neighbors * np.log(neighbors + 1e-12)).reshape((-1, 1))
    B_nx5 = np.dstack([neighbors,
                       neighbors * xx,
                       neighbors * yy,
                       neighbors * xx ** 2,
                       neighbors * yy ** 2]).reshape((-1, 5))

    # solving equation, read reference blog for details
    Q_nxn, R_nx5 = np.linalg.qr(B_nx5, mode='complete')
    S_5x1 = Q_nxn.T.dot(A_nx1)[:5]
    C_5x1 = np.linalg.inv(R_nx5[:5]).dot(S_5x1)

    # compute subpixel cx and cy from solved C
    cx = np.round(-C_5x1[1, 0] / (2 * C_5x1[3, 0]), decimals=5)
    cy = np.round(-C_5x1[2, 0] / (2 * C_5x1[4, 0]), decimals=5)

    return cx, cy


# Distribution-Aware Coordinate Representation 代码暂时正确
@DeprecationWarning
def calc_subpixel_gaussianCenter_dark(gaussianMap2d, center_xy_int, sigma_px=1):
    """
        :param gaussianMap2d: nxm高斯热图
        :param center_xy_int: 峰值点坐标
        :param sigma_px: 和训练一样的sigma参数,默认sigma xy方向相同
        :return: 中心点x,y坐标
        """
    assert isinstance(gaussianMap2d, np.ndarray)
    assert gaussianMap2d.ndim == 2
    center_xy_int = list(map(int, center_xy_int))
    # 求二阶导
    var_matrx = np.eye(2) * sigma_px * sigma_px
    hesshin = -np.linalg.inv(var_matrx)
    # 求一阶导,代码不正确,差分法结果也不对
    cv2_gaussianMap2d = gaussianMap2d.copy()
    uint8_gaussian_map_hw = (cv2_gaussianMap2d * 255.0).astype(np.uint8)
    grad_x = cv2.Sobel(uint8_gaussian_map_hw, cv2.CV_32F, 1, 0)  # 对x求一阶导
    grad_y = cv2.Sobel(uint8_gaussian_map_hw, cv2.CV_32F, 0, 1)  # 对y求一阶导
    grad_x = cv2.convertScaleAbs(grad_x) / 225.0
    grad_y = cv2.convertScaleAbs(grad_y) / 255.0
    gradx_int = grad_x[center_xy_int[1], center_xy_int[0]]
    grady_int = grad_y[center_xy_int[1], center_xy_int[0]]
    grad = np.array([gradx_int, grady_int]).reshape((2, 1))

    # 下面假装已经知道了gt值
    # center_xy = np.array(center_xy_int, np.float32).reshape((2, 1))
    # gt = np.array([33.356, 27.886], np.float32).reshape((2, 1))
    # grad = hesshin.dot(center_xy - gt)

    x0 = center_xy_int[0] + (var_matrx.dot(grad))[0]
    y0 = center_xy_int[1] + (var_matrx.dot(grad))[1]
    return round(x0[0], 5), round(y0[0], 5)


subpixel_method = {
    'paraboloid': calc_subpixel_gaussianCenter_paraboloid,
    'parabola': calc_subpixel_gaussianCenter_parabola,
    'empiric': calc_subpixel_gaussianCenter_empiric,
    'matrix': calc_subpixel_gaussianCenter_matrix,
    'dark': calc_subpixel_gaussianCenter_dark
}
