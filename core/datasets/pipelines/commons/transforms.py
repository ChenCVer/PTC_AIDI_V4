# -*- coding:utf-8 -*-
import cv2
import copy
import math
import mmcv
import random
import numpy as np
from .compose import Transform
from rraitools import ImageHelper
from ...registry import PIPELINES


__MAX_COORDS__ = [100000, 100000]


# TODO: 2020-11-09,后续好好研究下述实现的一系列Random_xxx数据增强系列.
__all__ = [
    'Normalize',
    'DeNormalize',
    'RandomBlur',
    'RandomNormalNoise',
    'RandomBrightContrast',
    'RandomHSV',
    'RandomHorizontallyFlip',
    'RandomVerticallyFlip',
    'RandomShift',
    'RandomRotate',
    'RandomScale',
    'RandomCrop',
    'RandomWarpPerspective',
    'RandomRegularCrop',
    'RegularCrop',
    'LetterResize',
]


@PIPELINES.register_module
class Normalize(object):
    """
    normalize a given img or a list of imgs.
    """

    def __init__(self, mean_bgr, std_bgr, to_rgb=True):
        """
        :param mean_bgr: list, like [0.0, 0.0, 0.0].
        :param std_bgr: list, like [1.0, 1.0, 1.0].
        :param to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
        """
        self.mean_bgr = np.array(mean_bgr, np.float32)
        self.std_bgr = np.array(std_bgr, np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        img = results.get('img', None)
        assert img is not None

        if not isinstance(img, list):
            if len(img.shape) == 2:
                assert self.mean_bgr.shape[0] == 1, 'mean dimensions do not match'
            else:
                assert img.shape[2] == self.mean_bgr.shape[0], 'mean dimensions do not match'
        else:
            if len(img[0].shape) == 2:
                assert self.mean_bgr.shape[0] == 1, 'mean dimensions do not match'
            else:
                assert img[0].shape[2] == self.mean_bgr.shape[0], 'mean dimensions do not match'

        if all(self.mean_bgr <= 1) and all(self.std_bgr <= 1):  # 归一化
            if not isinstance(img, list):
                # img = np.asarray(img).astype(np.float32) / 255.
                img_list = mmcv.imnormalize(img / 255., self.mean_bgr, self.std_bgr, self.to_rgb)
            else:
                img_list = []
                for _, data in enumerate(img):  # 考虑可能img size不固定
                    # img_data = np.asarray(data).astype(np.float32) / 255.
                    img_data = mmcv.imnormalize(img / 255., self.mean_bgr, self.std_bgr, self.to_rgb)
                    img_list.append(img_data)

            results['img'] = img_list
            results["img_shape"] = img.shape
            results['img_norm_cfg'] = dict(
                mean=self.mean_bgr, std=self.std_bgr, to_rgb=self.to_rgb)
            return results

        else:  # 标准化.
            if not isinstance(img, list):
                # img = (img - mean) / std
                img_list = mmcv.imnormalize(img, self.mean_bgr, self.std_bgr, self.to_rgb)
            else:
                img_list = []
                for _, data in enumerate(img):  # 考虑可能img size不固定
                    img_data = mmcv.imnormalize(img, self.mean_bgr, self.std_bgr, self.to_rgb)
                    img_list.append(img_data)

            results['img'] = img_list
            results["img_shape"] = img.shape
            results['img_norm_cfg'] = dict(
                mean=self.mean_bgr, std=self.std_bgr, to_rgb=False)

            return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean_bgr}, std={self.std_bgr}, to_rgb={self.to_rgb})'
        return repr_str


class DeNormalize(object):
    """
    normalize a given img or a list of imgs.
    """

    def __init__(self, mean_bgr, std_bgr, mode='BGR'):
        """
        :param mean: list, like [0.0, 0.0, 0.0].
        :param std: list, like [1.0, 1.0, 1.0].
        """
        self.mean_bgr = np.array(mean_bgr, np.float32)
        self.std_bgr = np.array(std_bgr, np.float32)
        if mode == 'RGB':
            self.mean_bgr = np.array(mean_bgr[::-1], np.float32)
            self.std_bgr = np.array(std_bgr[::-1], np.float32)

    def __call__(self, images):
        if not isinstance(images, list):
            if len(images.shape) == 2:
                assert self.mean_bgr.shape[0] == 1, 'mean dimensions do not match'
            else:
                assert images.shape[2] == self.mean_bgr.shape[0], 'mean dimensions do not match'
        else:
            if len(images[0].shape) == 2:
                assert self.mean_bgr.shape[0] == 1, 'mean dimensions do not match'
            else:
                assert images[0].shape[2] == self.mean_bgr.shape[0], 'mean dimensions do not match'

        if not isinstance(images, list):
            img = (images * self.std_bgr) + self.mean_bgr
            if all(self.mean_bgr <= 1) and all(self.std_bgr <= 1):
                img = img * 255.0
            return img.astype(np.uint8)
        else:
            img_list = []
            for _, data in enumerate(images):  # 考虑可能img size不固定
                img_data = (data * self.std_bgr) + self.mean_bgr
                if all(self.mean_bgr <= 1) and all(self.std_bgr <= 1):
                    img_data = img_data * 255.0
                img_list.append(img_data.astype(np.uint8))
            return img_list

# --------------------- 下述是自定义的数据增强操作 --------------------------- #


@PIPELINES.register_module()
class RandomBlur(Transform):  # 最好用双边模糊，因为可以保留边缘，看起来模糊效果更好
    """
    Randomly blur an image, better to use bilateral ambiguity.
    """

    def __init__(self, mode, prob, kernel_size_list):
        """
        :param parameter: [mode,probability,kernel_list], for example, [1, 0.5, [2, 3, 4, 5]].
        """
        super(RandomBlur, self).__init__(mode, prob)
        self.kernel_size_list = kernel_size_list  # blur kenerl 列表 exam=[2,3,4,5] kernel值越大，模糊越严重

    def get_parameter(self, image, **kwargs):
        rand_kernel_index = random.randint(0, len(self.kernel_size_list) - 1)
        size = (self.kernel_size_list[rand_kernel_index], self.kernel_size_list[rand_kernel_index])
        self.parameter_dict['size'] = size

    def apply_image(self, image, interpolation=cv2.INTER_LINEAR, **kwargs):
        size = self.parameter_dict['size']
        image = cv2.blur(image, size)
        return image


@PIPELINES.register_module()
class RandomNormalNoise(Transform):
    """
    Randomly add normal noise on an image.
    """

    def __init__(self, mode, prob, noise_scale):
        """
        :param parameter: mode,probability,noise_scale, for example, [1,0.5,5].
        """
        super(RandomNormalNoise, self).__init__(mode, prob)
        self.noise_scale = noise_scale  # exam=5, #噪声均值

    def get_parameter(self, image, **kwargs):
        assert len(image.shape) == 3
        num = image.shape[0] * image.shape[1] * image.shape[2]
        noise = np.random.normal(0, self.noise_scale, num)
        self.parameter_dict['noise'] = noise

    def apply_image(self, image, interpolation=cv2.INTER_LINEAR, **kwargs):
        noise = np.reshape(self.parameter_dict['noise'], image.shape)
        image = np.clip(np.float32(image) + np.float32(noise), 0, 255)
        return image.astype(np.uint8)


@PIPELINES.register_module()
class RandomBrightContrast(Transform):
    """
    Randoml change bright contrast of an image.
    """

    def __init__(self, mode, prob, alpha_scale=0.0, beta_scale=0):
        """
        :param parameter: [mode,probability,alpha_scale,beta_scale], for example, [1,0.5,0.2,20].
        """
        super(RandomBrightContrast, self).__init__(mode, prob)
        self.alpha_scale = alpha_scale
        self.beta_scale = beta_scale  # exam=0.2,20 g(x)=a*f(x)+b  a为对比度 b为亮度

    def get_parameter(self, image, **kwargs):
        alpha = self.alpha_scale * (random.random() * 2 - 1) + 1
        beta = (random.random() * 2 - 1) * self.beta_scale
        self.parameter_dict['alpha'] = alpha
        self.parameter_dict['beta'] = beta

    def apply_image(self, image, interpolation=cv2.INTER_LINEAR, **kwargs):
        return np.uint8(np.clip(np.float32(image) * self.parameter_dict['alpha'] + self.parameter_dict['beta'], 0, 255))


@PIPELINES.register_module()
class RandomHSV(Transform):
    """
    Randoml change hsv of an image.
    """

    def __init__(self, mode, prob, hue_scale=0, sat_scale=1., val_scale=1.):
        """
        :param parameter: [mode,probability,hue_scale,sat_scale,val_scale], for example, [1,0.5,0,1.5,1.5].
        """
        super(RandomHSV, self).__init__(mode, prob)
        self.hue_scale = hue_scale  # 0表示不起作用
        self.sat_scale = sat_scale  # 1表示不起作用
        self.val_scale = val_scale  # 1表示不起作用

    def get_parameter(self, image, **kwargs):
        hue = np.random.uniform(-self.hue_scale, self.hue_scale)
        sat = np.random.uniform(1, self.sat_scale) if random.random() < .5 else 1 / np.random.uniform(1, self.sat_scale)
        val = np.random.uniform(1, self.val_scale) if random.random() < .5 else 1 / np.random.uniform(1, self.val_scale)
        self.parameter_dict['hue'] = hue
        self.parameter_dict['sat'] = sat
        self.parameter_dict['val'] = val

    def apply_image(self, image, interpolation=cv2.INTER_LINEAR, **kwargs):
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image_hsv = np.array(image_hsv, np.float) / 255.0
        image_hsv[..., 0] += self.parameter_dict['hue']
        image_hsv[..., 0][image_hsv[..., 0] > 1] -= 1
        image_hsv[..., 0][image_hsv[..., 0] < 0] += 1
        image_hsv[..., 1] *= self.parameter_dict['sat']
        image_hsv[..., 2] *= self.parameter_dict['val']
        image_hsv[image_hsv > 1] = 1
        image_hsv[image_hsv < 0] = 0
        image_hsv = np.uint8(image_hsv * 255.0)
        image_bgr = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)
        return image_bgr


@PIPELINES.register_module()
class RandomHorizontallyFlip(Transform):
    def __init__(self, mode, prob):
        """
        :param parameter: [mode,probability], for example, [1, 0.5].
        """
        super(RandomHorizontallyFlip, self).__init__(mode, prob)

    def get_parameter(self, image, **kwargs):
        _, w = image.shape[:2]
        self.parameter_dict['w'] = w

    def apply_image(self, image, interpolation=cv2.INTER_LINEAR, **kwargs):
        return cv2.flip(image, 1)

    def apply_segmentation(self, mask, **kwargs):
        return self.apply_image(mask)

    def apply_box(self, box_xyxy, **kwargs):
        assert isinstance(box_xyxy, np.ndarray)
        box_xyxy = copy.deepcopy(box_xyxy)
        xmin = self.parameter_dict['w'] - 1 - box_xyxy[:, 2]
        xmax = self.parameter_dict['w'] - 1 - box_xyxy[:, 0]
        box_xyxy[:, 0] = xmin
        box_xyxy[:, 2] = xmax
        return box_xyxy.astype(np.int64)

    def apply_coords(self, coords_nxy, **kwargs):
        assert isinstance(coords_nxy, np.ndarray)
        coords_nxy = copy.deepcopy(coords_nxy)
        coords_nxy[:, 0] = self.parameter_dict['w'] - 1 - coords_nxy[:, 0]
        return coords_nxy.astype(np.float32)


@PIPELINES.register_module()
class RandomVerticallyFlip(Transform):  # [mode,probability]
    def __init__(self, mode, prob):
        """
        :param parameter: [mode,probability], for example, [1, 0.5].
        """
        super(RandomVerticallyFlip, self).__init__(mode, prob)

    def get_parameter(self, image, **kwargs):
        h, _ = image.shape[:2]
        self.parameter_dict['h'] = h

    def apply_image(self, image, interpolation=cv2.INTER_LINEAR, **kwargs):
        return cv2.flip(image, 0)

    def apply_segmentation(self, mask, **kwargs):
        return self.apply_image(mask)

    def apply_box(self, box_xyxy, **kwargs):
        assert isinstance(box_xyxy, np.ndarray)
        box_xyxy = copy.deepcopy(box_xyxy)
        ymin = self.parameter_dict['h'] - 1 - box_xyxy[:, 3]
        ymax = self.parameter_dict['h'] - 1 - box_xyxy[:, 1]
        box_xyxy[:, 1] = ymin
        box_xyxy[:, 3] = ymax
        return box_xyxy.astype(np.int64)

    def apply_coords(self, coords_nxy, **kwargs):
        assert isinstance(coords_nxy, np.ndarray)
        coords_nxy = copy.deepcopy(coords_nxy)
        coords_nxy[:, 1] = self.parameter_dict['h'] - 1 - coords_nxy[:, 1]
        return coords_nxy.astype(np.float32)


@PIPELINES.register_module()
class RandomShift(Transform):  # 平移后图片size不变
    def __init__(self, mode, prob, shift_range=(0, 0), border_value=0):
        """
        Random shift will not change image size.

        :param parameter: [mode,probability,shift_pix_range_hw], for example, [1,0.5,[10,10]].
        """
        super(RandomShift, self).__init__(mode, prob)  # [1,0.5,[10,10]]
        self.shift_pix_range_hw = shift_range
        self.border_value = border_value

    def get_parameter(self, image, **kwargs):
        random_shift_x = random.randint(-self.shift_pix_range_hw[1], self.shift_pix_range_hw[1])
        random_shft_y = random.randint(-self.shift_pix_range_hw[0], self.shift_pix_range_hw[0])
        shift_matrix = np.float32([[1, 0, random_shift_x],
                                   [0, 1, random_shft_y]])
        self.parameter_dict['shift_matrix'] = shift_matrix
        self.parameter_dict['shape_hw'] = image.shape[:2]

    def apply_image(self, image, interpolation=cv2.INTER_LINEAR, **kwargs):
        return cv2.warpAffine(image, self.parameter_dict['shift_matrix'],
                              self.parameter_dict['shape_hw'][::-1],
                              flags=interpolation,
                              borderValue=self.border_value)

    def apply_segmentation(self, mask, **kwargs):
        return self.apply_image(mask, cv2.INTER_NEAREST).astype(np.uint8)

    def apply_box(self, box_xyxy, **kwargs):
        assert isinstance(box_xyxy, np.ndarray)
        box_xyxy = copy.deepcopy(box_xyxy)
        box_xyxy_out = []
        for bbox in box_xyxy:
            class_id = bbox[4]
            # 4个顶点坐标，变换后可能不是矩形
            bbox_temp = [bbox[0], bbox[1], bbox[2], bbox[1],
                         bbox[0], bbox[3], bbox[2], bbox[3]]

            for node in range(4):
                x = bbox_temp[node * 2]
                y = bbox_temp[node * 2 + 1]
                p = np.array([x, y, 1])
                p = self.parameter_dict['shift_matrix'].dot(p)
                bbox_temp[node * 2] = p[0]
                bbox_temp[node * 2 + 1] = p[1]

            temp = [min(bbox_temp[0], bbox_temp[2], bbox_temp[4], bbox_temp[6]),
                    min(bbox_temp[1], bbox_temp[3], bbox_temp[5], bbox_temp[7]),
                    max(bbox_temp[0], bbox_temp[2], bbox_temp[4], bbox_temp[6]),
                    max(bbox_temp[1], bbox_temp[3], bbox_temp[5], bbox_temp[7]),
                    class_id]
            # box可能出界
            temp[0] = max(temp[0], 0)
            temp[1] = max(temp[1], 0)
            temp[2] = min(temp[2], self.parameter_dict['shape_hw'][1])
            temp[3] = min(temp[3], self.parameter_dict['shape_hw'][0])
            # 如果bbox完全出界，则抛弃
            if temp[2] <= 0 or temp[3] <= 0 or \
                    temp[0] >= self.parameter_dict['shape_hw'][1] or \
                    temp[1] >= self.parameter_dict['shape_hw'][0]:
                continue
            box_xyxy_out.append(temp)
        return np.array(box_xyxy_out).astype(np.int64)

    def apply_coords(self, coords_nxy, **kwargs):
        assert isinstance(coords_nxy, np.ndarray)
        coords_nxy = copy.deepcopy(coords_nxy)
        rotation_matrix = self.parameter_dict['shift_matrix'][:, :2]
        translation = np.array([self.parameter_dict['shift_matrix'][0][2],
                                self.parameter_dict['shift_matrix'][1][2]])
        coords_nxy = np.dot(rotation_matrix, coords_nxy.T).T + translation
        # 越界的关键点需要赋予不可能的值,不可以抛弃该关键点
        index = (coords_nxy[:, 0] < 0) | (coords_nxy[:, 1] < 0) | \
                (coords_nxy[:, 0] >= self.parameter_dict['shape_hw'][1]) | \
                (coords_nxy[:, 1] >= self.parameter_dict['shape_hw'][0])
        coords_nxy[index] = np.array(__MAX_COORDS__)
        return coords_nxy.astype(np.float32)


@PIPELINES.register_module()
class RandomRotate(Transform):  # 为了不丢失信息，可能图片会放大
    def __init__(self, mode, prob, angle_scale=0, pts_offset=0):  # [model,probability,angle,pts_offset]
        """
        Image size may be enlarged to remain information.

        :param parameter: [model,probability,angle,pts_offset], for example, [1,0.5,30,10].

        """
        super(RandomRotate, self).__init__(mode, prob)  # exam=[1,0.5,30,10]
        self.angle = angle_scale  # 旋转角度
        self.pts_offset = pts_offset  # 相对于图片中心偏移像素

    def get_parameter(self, image, **kwargs):
        height, width = image.shape[:2]
        # 如果输入是tuple或者list，则角度是给定的任意值
        if isinstance(self.angle, (tuple, list)):
            random_angle = random.choice(self.angle)
        else:
            random_angle = random.randint(-self.angle, self.angle)
        random_offset = random.randint(-self.pts_offset, self.pts_offset)
        angle_pts_xy = (random_offset + image.shape[1] // 2, random_offset + image.shape[0] // 2)
        rotate_mat = cv2.getRotationMatrix2D(angle_pts_xy, random_angle, 1.0)
        cos_val = np.abs(rotate_mat[0, 0])
        sin_val = np.abs(rotate_mat[0, 1])
        new_width = int(height * sin_val + width * cos_val)
        new_height = int(height * cos_val + width * sin_val)
        rotate_mat[0, 2] += (new_width / 2.) - angle_pts_xy[0]
        rotate_mat[1, 2] += (new_height / 2.) - angle_pts_xy[1]
        self.parameter_dict['rotate_mat'] = rotate_mat
        self.parameter_dict['new_shape_wh'] = (new_width, new_height)

    def apply_image(self, image, interpolation=cv2.INTER_LINEAR, **kwargs):
        return cv2.warpAffine(image, self.parameter_dict['rotate_mat'], self.parameter_dict['new_shape_wh'],
                              flags=interpolation)

    def apply_segmentation(self, mask, **kwargs):
        return self.apply_image(mask, cv2.INTER_NEAREST).astype(np.uint8)

    def apply_box(self, box_xyxy, **kwargs):
        assert isinstance(box_xyxy, np.ndarray)
        box_xyxy_out = []
        for bbox in box_xyxy:
            class_id = bbox[4]
            # 4个顶点坐标，变换后可能不是矩形
            bbox_temp = [bbox[0], bbox[1], bbox[2], bbox[1],
                         bbox[0], bbox[3], bbox[2], bbox[3]]

            for node in range(4):
                x = bbox_temp[node * 2]
                y = bbox_temp[node * 2 + 1]
                p = np.array([x, y, 1])
                p = self.parameter_dict['rotate_mat'].dot(p)
                bbox_temp[node * 2] = p[0]
                bbox_temp[node * 2 + 1] = p[1]

            temp = [min(bbox_temp[0], bbox_temp[2], bbox_temp[4], bbox_temp[6]),
                    min(bbox_temp[1], bbox_temp[3], bbox_temp[5], bbox_temp[7]),
                    max(bbox_temp[0], bbox_temp[2], bbox_temp[4], bbox_temp[6]),
                    max(bbox_temp[1], bbox_temp[3], bbox_temp[5], bbox_temp[7]),
                    class_id]
            # box可能出界
            temp[0] = max(temp[0], 0)
            temp[1] = max(temp[1], 0)
            temp[2] = min(temp[2], self.parameter_dict['new_shape_wh'][0])
            temp[3] = min(temp[3], self.parameter_dict['new_shape_wh'][1])
            # 如果bbox完全出界，则抛弃
            if temp[2] <= 0 or temp[3] <= 0 or \
                    temp[0] >= self.parameter_dict['new_shape_wh'][0] or \
                    temp[1] >= self.parameter_dict['new_shape_wh'][1]:
                continue
            box_xyxy_out.append(temp)
        return np.array(box_xyxy_out).astype(np.int64)

    def apply_coords(self, coords_nxy, **kwargs):
        assert isinstance(coords_nxy, np.ndarray)
        coords_nxy = copy.deepcopy(coords_nxy)
        rotation_matrix = self.parameter_dict['rotate_mat'][:, :2]
        translation = np.array([self.parameter_dict['rotate_mat'][0][2], self.parameter_dict['rotate_mat'][1][2]])
        coords_nxy = np.dot(rotation_matrix, coords_nxy.T).T + translation
        # 越界的关键点需要赋予不可能的值,不可以抛弃该关键点
        index = (coords_nxy[:, 0] < 0) | (coords_nxy[:, 1] < 0) | \
                (coords_nxy[:, 0] >= self.parameter_dict['new_shape_wh'][0]) | \
                (coords_nxy[:, 1] >= self.parameter_dict['new_shape_wh'][1])
        coords_nxy[index] = np.array(__MAX_COORDS__)
        return coords_nxy.astype(np.float32)


@PIPELINES.register_module()
class RandomScale(Transform):
    def __init__(self, mode, prob, scale_range=(1.0, 1.0), aspect_range=(1.0, 1.0)):
        super(RandomScale, self).__init__(mode, prob)
        self._scale_range = scale_range
        self._aspect_range = aspect_range

    def get_parameter(self, image, **kwargs):
        height, width = image.shape[:2]
        scale_ratio = random.uniform(*self._scale_range)
        aspect_ratio = random.uniform(*self._aspect_range)
        w_scale_ratio = math.sqrt(aspect_ratio) * scale_ratio
        h_scale_ratio = math.sqrt(1.0 / aspect_ratio) * scale_ratio
        self.parameter_dict['scale_ratio_wh'] = (w_scale_ratio, h_scale_ratio)
        self.parameter_dict['converted_size_wh'] = (int(width * self.parameter_dict['scale_ratio_wh'][0]),
                                                    int(height * self.parameter_dict['scale_ratio_wh'][1]))

    def apply_image(self, image, interpolation=cv2.INTER_LINEAR, **kwargs):
        return cv2.resize(image, self.parameter_dict['converted_size_wh'], interpolation=interpolation).astype(
            np.uint8)

    def apply_segmentation(self, mask, **kwargs):
        return self.apply_image(mask, cv2.INTER_NEAREST)

    def apply_box(self, box_xyxy, **kwargs):
        box_xyxy = copy.deepcopy(box_xyxy)
        box_xyxy = np.array(box_xyxy, np.float32)
        box_xyxy[:, [0, 2]] *= np.ones_like(box_xyxy[:, [0, 2]], np.float32) * self.parameter_dict['scale_ratio_wh'][0]
        box_xyxy[:, [1, 3]] *= np.ones_like(box_xyxy[:, [1, 3]], np.float32) * self.parameter_dict['scale_ratio_wh'][1]
        return box_xyxy.astype(np.int64)  # 不可能会出界，故不用判断

    def apply_coords(self, coords_nxy, **kwargs):
        coords_nxy = copy.deepcopy(coords_nxy)
        coords_nxy = np.array(coords_nxy, np.float32)
        coords_nxy[:, 0] *= self.parameter_dict['scale_ratio_wh'][0]
        coords_nxy[:, 1] *= self.parameter_dict['scale_ratio_wh'][1]
        return coords_nxy.astype(np.float32)


@PIPELINES.register_module()
class RandomCrop(Transform):
    def __init__(self, mode, prob, pix_range_hw=(0.0, 0.0)):
        """
        :param parameter: [mode,probability,crop_pix_range_hw], example=[1, 0.5, [10, 10]].
        """
        super(RandomCrop, self).__init__(mode, prob)
        self._crop_pix_range_hw = pix_range_hw

    def get_parameter(self, image, **kwargs):
        h, w = image.shape[:-1]
        x1 = random.randint(0, self._crop_pix_range_hw[1] // 2)
        y1 = random.randint(0, self._crop_pix_range_hw[0] // 2)
        x2 = random.randint(0, self._crop_pix_range_hw[1] // 2)
        y2 = random.randint(0, self._crop_pix_range_hw[0] // 2)
        self.parameter_dict['crop_para'] = [x1, y1, w - 1 - x2, h - 1 - y2]

    def apply_image(self, image, interpolation=cv2.INTER_LINEAR, **kwargs):
        [x1, y1, x2, y2] = self.parameter_dict['crop_para']
        assert x2 > x1 and y2 > y1, 'unreasonable parameter setting'
        return image[y1:y2, x1:x2]

    def apply_segmentation(self, mask, **kwargs):
        return self.apply_image(mask)

    def apply_box(self, box_xyxy, **kwargs):
        box_xyxy = copy.deepcopy(box_xyxy)
        box_xyxy = np.array(box_xyxy, np.float32)
        [x1, y1, x2, y2] = self.parameter_dict['crop_para']
        w = x2 - x1
        h = y2 - y1
        box_xyxy[:, [0, 2]] -= np.ones_like(box_xyxy[:, [0, 2]], np.float32) * x1
        box_xyxy[:, [1, 3]] -= np.ones_like(box_xyxy[:, [1, 3]], np.float32) * y1
        # 可能出界
        box_xyxy[:, 0:2][box_xyxy[:, 0:2] < 0] = 0
        box_xyxy[:, 2][box_xyxy[:, 2] > w] = w
        box_xyxy[:, 3][box_xyxy[:, 3] > h] = h
        index = (box_xyxy[:, 2] >= 0) & (box_xyxy[:, 1] >= 0) & \
                (box_xyxy[:, 0] < w) & \
                (box_xyxy[:, 1] < h)
        return box_xyxy[index].astype(np.int64)

    def apply_coords(self, coords_nxy, **kwargs):
        coords_nxy = copy.deepcopy(coords_nxy)
        [x1, y1, x2, y2] = self.parameter_dict['crop_para']
        w = x2 - x1
        h = y2 - y1
        coords_nxy = np.array(coords_nxy, np.float32)
        coords_nxy[:, 0] -= x1
        coords_nxy[:, 1] -= y1
        # 越界的关键点需要赋予不可能的值,不可以抛弃该关键点
        index = (coords_nxy[:, 0] < 0) | (coords_nxy[:, 1] < 0) | \
                (coords_nxy[:, 0] >= w) | \
                (coords_nxy[:, 1] >= h)
        coords_nxy[index] = np.array(__MAX_COORDS__)
        return coords_nxy.astype(np.float32)


@PIPELINES.register_module()
class RegularCrop(Transform):  # 保证输出一定是相同大小
    def __init__(self, mode, prob, crop_size_hw):
        """
        Regular crop will return an image with same size of input image.

        :param parameter: [mode,probability,crop_shape_hw], example=[1, 0.5, [256, 256]].
        """
        super(RegularCrop, self).__init__(mode, prob)
        self._crop_size_hw = crop_size_hw

    def get_parameter(self, image, **kwargs):
        h, w = image.shape[:-1]
        # 如果输出比输入小，则需要pad
        pad_w = max(self._crop_size_hw[1] - w, 0)
        pad_h = max(self._crop_size_hw[0] - h, 0)
        top = pad_h // 2
        left = pad_w // 2
        bottom = pad_h - top
        right = pad_w - left
        # 如果输出比输入大，则需要crop
        x = max(w - self._crop_size_hw[1], 0)
        y = max(h - self._crop_size_hw[0], 0)
        self.parameter_dict['pad_para'] = [top, bottom, left, right]
        self.parameter_dict['crop_para'] = [x, y]

    def apply_image(self, image, interpolation=cv2.INTER_LINEAR, **kwargs):
        image = ImageHelper.pad_img(image, self.parameter_dict['pad_para'][0], self.parameter_dict['pad_para'][1],
                                    self.parameter_dict['pad_para'][2], self.parameter_dict['pad_para'][3], (0, 0, 0))
        y = self.parameter_dict['crop_para'][1]
        x = self.parameter_dict['crop_para'][0]
        return image[y // 2:y // 2 + self._crop_size_hw[0], x // 2:x // 2 + self._crop_size_hw[1]]  # 居中裁剪

    def apply_segmentation(self, mask, **kwargs):
        return self.apply_image(mask).astype(np.uint8)

    def apply_box(self, box_xyxy, **kwargs):
        box_xyxy = copy.deepcopy(box_xyxy)
        box_xyxy = np.array(box_xyxy, np.float32)
        left = self.parameter_dict['pad_para'][2]
        top = self.parameter_dict['pad_para'][0]
        x = self.parameter_dict['crop_para'][0]
        y = self.parameter_dict['crop_para'][1]
        box_xyxy[:, [0, 2]] += np.ones_like(box_xyxy[:, [0, 2]], np.float32) * left
        box_xyxy[:, [1, 3]] += np.ones_like(box_xyxy[:, [1, 3]], np.float32) * top
        box_xyxy[:, [0, 2]] -= np.ones_like(box_xyxy[:, [0, 2]], np.float32) * (x / 2)
        box_xyxy[:, [1, 3]] -= np.ones_like(box_xyxy[:, [1, 3]], np.float32) * (y / 2)
        # 可能出界
        box_xyxy[:, 0:2][box_xyxy[:, 0:2] < 0] = 0
        box_xyxy[:, 2][box_xyxy[:, 2] > self._crop_size_hw[1]] = self._crop_size_hw[1]
        box_xyxy[:, 3][box_xyxy[:, 3] > self._crop_size_hw[0]] = self._crop_size_hw[0]
        index = (box_xyxy[:, 2] >= 0) & (box_xyxy[:, 1] >= 0) & \
                (box_xyxy[:, 0] < self._crop_size_hw[1]) & \
                (box_xyxy[:, 1] < self._crop_size_hw[0])
        return box_xyxy[index].astype(np.int64)

    def apply_coords(self, coords_nxy, **kwargs):
        assert isinstance(coords_nxy, np.ndarray)
        coords_nxy = copy.deepcopy(coords_nxy)
        left = self.parameter_dict['pad_para'][2]
        top = self.parameter_dict['pad_para'][0]
        x = self.parameter_dict['crop_para'][0]
        y = self.parameter_dict['crop_para'][1]
        coords_nxy = np.array(coords_nxy, np.float32)
        coords_nxy[:, 0] += left
        coords_nxy[:, 1] += top
        coords_nxy[:, 0] -= x / 2
        coords_nxy[:, 1] -= y / 2
        # 越界的关键点需要赋予不可能的值,不可以抛弃该关键点
        index = (coords_nxy[:, 0] < 0) | (coords_nxy[:, 1] < 0) | \
                (coords_nxy[:, 0] >= self._crop_size_hw[1]) | \
                (coords_nxy[:, 1] >= self._crop_size_hw[0])
        coords_nxy[index] = np.array(__MAX_COORDS__)
        return coords_nxy.astype(np.float32)


# 随机透视变换(不保持平行关系)：4点法
@PIPELINES.register_module()
class RandomWarpPerspective(Transform):  # [model,probability,warp_pix]
    def __init__(self, mode, prob, warp_pix_scale):
        """
        :param parameter: [mode,probability,warp_pix], example=[1,0.5,10].
        """
        super(RandomWarpPerspective, self).__init__(mode, prob)
        self._warp_pix = warp_pix_scale  # 透视范围 exam=[1,0.5,10]

    def get_parameter(self, image, **kwargs):
        h, w = image.shape[:-1]
        x0 = random.randint(-self._warp_pix, self._warp_pix)
        y0 = random.randint(-self._warp_pix, self._warp_pix)
        x1 = random.randint(w - self._warp_pix, w + self._warp_pix)
        y1 = random.randint(-self._warp_pix, self._warp_pix)
        x2 = random.randint(-self._warp_pix, self._warp_pix)
        y2 = random.randint(h - self._warp_pix, h + self._warp_pix)
        x3 = random.randint(w - self._warp_pix, w + self._warp_pix)
        y3 = random.randint(h - self._warp_pix, h + self._warp_pix)
        src_list_xy = [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]
        dst_list_xy = [[0, 0], [w, 0], [0, h], [w, h]]
        _, perspective_matrix = self.__quad_2_rect(image, src_list_xy, dst_list_xy, (w, h), flags=cv2.INTER_LINEAR)
        self.parameter_dict['src_list_xy'] = src_list_xy
        self.parameter_dict['dst_list_xy'] = dst_list_xy
        self.parameter_dict['perspective_matrix'] = perspective_matrix

    def apply_image(self, image, interpolation=cv2.INTER_LINEAR, **kwargs):
        h, w = image.shape[:-1]
        image, _ = self.__quad_2_rect(image, self.parameter_dict['src_list_xy'], self.parameter_dict['dst_list_xy'],
                                      (w, h), flags=interpolation)
        return image

    def apply_segmentation(self, mask, **kwargs):
        return self.apply_image(mask, cv2.INTER_NEAREST).astype(np.uint8)

    def apply_box(self, box_xyxy, **kwargs):
        assert isinstance(box_xyxy, np.ndarray)
        box_xyxy = copy.deepcopy(box_xyxy)
        w = self.parameter_dict['dst_list_xy'][3][0]
        h = self.parameter_dict['dst_list_xy'][3][1]
        box_xyxy_out = []
        for bbox in box_xyxy:
            # 4个顶点坐标，变换后可能不是矩形
            class_id = bbox[4]
            bbox_temp = [bbox[0], bbox[1], bbox[2], bbox[1],
                         bbox[0], bbox[3], bbox[2], bbox[3]]

            for node in range(4):
                x = bbox_temp[node * 2]
                y = bbox_temp[node * 2 + 1]
                temp_box_xyxy = np.array([x, y], dtype='float32')
                temp_box_xyxy = np.expand_dims(temp_box_xyxy, axis=0)
                temp_box_xyxy = np.expand_dims(temp_box_xyxy, axis=0)
                temp_box_xyxy = cv2.perspectiveTransform(temp_box_xyxy, self.parameter_dict['perspective_matrix'])
                temp_box_xyxy = np.squeeze(temp_box_xyxy)
                bbox_temp[node * 2] = temp_box_xyxy[0]
                bbox_temp[node * 2 + 1] = temp_box_xyxy[1]

            temp = [min(bbox_temp[0], bbox_temp[2], bbox_temp[4], bbox_temp[6]),
                    min(bbox_temp[1], bbox_temp[3], bbox_temp[5], bbox_temp[7]),
                    max(bbox_temp[0], bbox_temp[2], bbox_temp[4], bbox_temp[6]),
                    max(bbox_temp[1], bbox_temp[3], bbox_temp[5], bbox_temp[7]),
                    class_id]
            # box可能出界
            temp[0] = max(temp[0], 0)
            temp[1] = max(temp[1], 0)
            temp[2] = min(temp[2], w)
            temp[3] = min(temp[3], h)
            # 如果bbox完全出界，则抛弃
            if temp[2] <= 0 or temp[3] <= 0 or \
                    temp[0] >= w or \
                    temp[1] >= h:
                continue
            box_xyxy_out.append(temp)
        return np.array(box_xyxy_out, np.int64)

    def apply_coords(self, coords_nxy, **kwargs):
        assert isinstance(coords_nxy, np.ndarray)
        coords_nxy = copy.deepcopy(coords_nxy)
        w = self.parameter_dict['dst_list_xy'][3][0]
        h = self.parameter_dict['dst_list_xy'][3][1]
        coords_nxy = np.array([coords_nxy], dtype='float32')
        coords_nxy = cv2.perspectiveTransform(coords_nxy, self.parameter_dict['perspective_matrix'])
        coords_nxy = coords_nxy[0, ...]
        index = (coords_nxy[:, 0] >= 0) & (coords_nxy[:, 1] >= 0) & (coords_nxy[:, 0] < w) & (coords_nxy[:, 1] < h)
        coords_nxy = coords_nxy[index]
        return coords_nxy.astype(np.float32)

    def __quad_2_rect(self, img, src_list_xy, dst_size_xy, dst_size, flags=cv2.INTER_NEAREST):
        pts_src = np.float32(src_list_xy)
        pts_dst = np.float32(dst_size_xy)
        M = cv2.getPerspectiveTransform(pts_src, pts_dst)
        dst = cv2.warpPerspective(img, M, dst_size, flags=flags)
        return dst, M


@PIPELINES.register_module()
class LetterResize(Transform):  # 不丢失任何信息的resize操作
    def __init__(self, mode, prob, shape_hw, interpolation=cv2.INTER_LINEAR):
        super().__init__(mode, prob)
        self._resize_size_hw = shape_hw
        self._interpolation = interpolation

    def get_parameter(self, image, **kwargs):
        h, w = image.shape[:2]
        if w == self._resize_size_hw[1] and h == self._resize_size_hw[0]:
            scale = 1.0
            pad = (0, 0, 0, 0)
        else:
            if w / self._resize_size_hw[1] >= h / self._resize_size_hw[0]:
                scale = self._resize_size_hw[1] / w
            else:
                scale = self._resize_size_hw[0] / h
            new_w = int(w * scale)
            new_h = int(h * scale)
            if new_w == self._resize_size_hw[1] and new_h == self._resize_size_hw[0]:
                pad = (0, 0, 0, 0)
            else:
                pad_w = (self._resize_size_hw[1] - new_w) / 2
                pad_h = (self._resize_size_hw[0] - new_h) / 2
                pad = (int(pad_w), int(pad_h), int(pad_w + .5), int(pad_h + .5))
        self.parameter_dict['scale'] = scale
        self.parameter_dict['pad_tblr'] = pad

    def apply_image(self, image, interpolation=cv2.INTER_LINEAR, **kwargs):
        h, w = image.shape[:2]
        if self.parameter_dict['scale'] != 1:
            new_w = int(w * self.parameter_dict['scale'])
            new_h = int(h * self.parameter_dict['scale'])
            image = cv2.resize(image, (new_w, new_h), interpolation=self._interpolation)
        top = self.parameter_dict['pad_tblr'][1]
        bottom = self.parameter_dict['pad_tblr'][3]
        left = self.parameter_dict['pad_tblr'][0]
        right = self.parameter_dict['pad_tblr'][2]
        return ImageHelper.pad_img(image, top, bottom, left, right, (0, 0, 0))

    def apply_segmentation(self, mask, **kwargs):
        return self.apply_image(mask, cv2.INTER_NEAREST)

    def apply_box(self, box_xyxy, **kwargs):
        box_xyxy = copy.deepcopy(box_xyxy)
        box_xyxy = np.array(box_xyxy, np.float32)
        if self.parameter_dict['scale'] != 1:
            box_xyxy[:, :4] *= np.ones_like(box_xyxy[:, :4], np.float32) * self.parameter_dict['scale']
        top = self.parameter_dict['pad_tblr'][1]
        left = self.parameter_dict['pad_tblr'][0]
        box_xyxy[:, [0, 2]] += np.ones_like(box_xyxy[:, [0, 2]], np.float32) * left
        box_xyxy[:, [1, 3]] += np.ones_like(box_xyxy[:, [1, 3]], np.float32) * top
        return box_xyxy.astype(np.int64)  # TODO 实际上最好不要int，只不过有些数据标注出界后面会报错

    def apply_coords(self, coords_nxy, **kwargs):
        coords_nxy = copy.deepcopy(coords_nxy)
        coords_nxy = np.array(coords_nxy, np.float32)
        if self.parameter_dict['scale'] != 1:
            coords_nxy *= self.parameter_dict['scale']
        top = self.parameter_dict['pad_tblr'][1]
        left = self.parameter_dict['pad_tblr'][0]
        coords_nxy[:, 0] += left
        coords_nxy[:, 1] += top
        return coords_nxy


@PIPELINES.register_module()
class RandomRegularCrop(Transform):  # 保证输出一定是相同大小
    def __init__(self, mode, prob, shape_hw):
        """
        Regular crop will return an image with same size of input image.
        :param parameter: [mode,probability,crop_shape_hw], example=[1, 0.5, [256, 256]].
        """
        super().__init__(mode, prob)
        self._crop_size_hw = shape_hw

    def get_parameter(self, image, **kwargs):
        h, w = image.shape[:2]
        # 如果输出比输入小，则需要pad
        pad_w = max(self._crop_size_hw[1] - w, 0)
        pad_h = max(self._crop_size_hw[0] - h, 0)
        top = pad_h // 2
        left = pad_w // 2
        bottom = pad_h - top
        right = pad_w - left
        # 如果输出比输入大，则需要crop
        x = random.randint(0, max(w - self._crop_size_hw[1], 0))
        y = random.randint(0, max(h - self._crop_size_hw[0], 0))
        center_xy = (x + self._crop_size_hw[1] // 2, y + self._crop_size_hw[0] // 2)
        self.parameter_dict['pad_para'] = [top, bottom, left, right]
        self.parameter_dict['crop_para'] = center_xy

    def apply_image(self, image, interpolation=cv2.INTER_LINEAR, **kwargs):
        image = ImageHelper.pad_img(image, self.parameter_dict['pad_para'][0], self.parameter_dict['pad_para'][1],
                                    self.parameter_dict['pad_para'][2], self.parameter_dict['pad_para'][3], (0, 0, 0))
        x = self.parameter_dict['crop_para'][0]
        y = self.parameter_dict['crop_para'][1]
        img = image[y - self._crop_size_hw[0] // 2:y - self._crop_size_hw[0] // 2 + self._crop_size_hw[0],
              x - self._crop_size_hw[1] // 2:x - self._crop_size_hw[1] // 2 + self._crop_size_hw[1]]
        return img

    def apply_segmentation(self, mask, **kwargs):
        return self.apply_image(mask).astype(np.uint8)

    # def apply_box(self, box_xyxy, **kwargs):
    #     box_xyxy = copy.deepcopy(box_xyxy)
    #     box_xyxy = np.array(box_xyxy, np.float32)
    #     left = self.parameter_dict['pad_para'][2]
    #     top = self.parameter_dict['pad_para'][0]
    #     x = self.parameter_dict['crop_para'][0]
    #     y = self.parameter_dict['crop_para'][1]
    #     box_xyxy[:, [0, 2]] += np.ones_like(box_xyxy[:, [0, 2]], np.float32) * left
    #     box_xyxy[:, [1, 3]] += np.ones_like(box_xyxy[:, [1, 3]], np.float32) * top
    #     box_xyxy[:, [0, 2]] -= np.ones_like(box_xyxy[:, [0, 2]], np.float32) * (x / 2)
    #     box_xyxy[:, [1, 3]] -= np.ones_like(box_xyxy[:, [1, 3]], np.float32) * (y / 2)
    #     # 可能出界
    #     box_xyxy[:, 0:2][box_xyxy[:, 0:2] < 0] = 0
    #     box_xyxy[:, 2][box_xyxy[:, 2] > self._crop_size_hw[1]] = self._crop_size_hw[1]
    #     box_xyxy[:, 3][box_xyxy[:, 3] > self._crop_size_hw[0]] = self._crop_size_hw[0]
    #     index = (box_xyxy[:, 2] >= 0) & (box_xyxy[:, 1] >= 0) & \
    #             (box_xyxy[:, 0] < self._crop_size_hw[1]) & \
    #             (box_xyxy[:, 1] < self._crop_size_hw[0])
    #     return box_xyxy[index].astype(np.int64)
    #
    # def apply_coords(self, coords_nxy, **kwargs):
    #     assert isinstance(coords_nxy, np.ndarray)
    #     coords_nxy = copy.deepcopy(coords_nxy)
    #     left = self.parameter_dict['pad_para'][2]
    #     top = self.parameter_dict['pad_para'][0]
    #     x = self.parameter_dict['crop_para'][0]
    #     y = self.parameter_dict['crop_para'][1]
    #     coords_nxy = np.array(coords_nxy, np.float32)
    #     coords_nxy[:, 0] += left
    #     coords_nxy[:, 1] += top
    #     coords_nxy[:, 0] -= x / 2
    #     coords_nxy[:, 1] -= y / 2
    #     # 越界的关键点需要赋予不可能的值,不可以抛弃该关键点
    #     index = (coords_nxy[:, 0] < 0) | (coords_nxy[:, 1] < 0) | \
    #             (coords_nxy[:, 0] >= self._crop_size_hw[1]) | \
    #             (coords_nxy[:, 1] >= self._crop_size_hw[0])
    #     coords_nxy[index] = np.array(__MAX_COORDS__)
    #     return coords_nxy.astype(np.float32)
