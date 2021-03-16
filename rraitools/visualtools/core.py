import numpy as np
import cv2
import torch.nn.functional as F
from ..imgtools import ImageHelper


class VisualHelper(object):
    @staticmethod
    # 可视化显示相关
    def show_bbox(image, bboxs_list, color=None,
                  thickness=1, font_scale=0.3, wait_time_ms=0, names=None,
                  is_show=True, is_without_mask=False):
        """
        Visualize bbox in object detection by drawing rectangle.

        :param image: numpy.ndarray.
        :param bboxs_list: list: [pts_xyxy, prob, id]: label or prediction.
        :param color: tuple.
        :param thickness: int.
        :param fontScale: float.
        :param wait_time_ms: int
        :param names: string: window name
        :param is_show: bool: whether to display during middle process
        :return: numpy.ndarray
        """
        from ..visualtools import random_color
        assert image is not None
        font = cv2.FONT_HERSHEY_SIMPLEX
        image_copy = image.copy()
        for bbox in bboxs_list:
            if len(bbox) == 5:
                txt = '{:.3f}'.format(bbox[4])
            elif len(bbox) == 6:
                txt = 'p={:.3f},id={:.3f}'.format(bbox[4], bbox[5])
            bbox_f = np.array(bbox[:4], np.int32)
            if color is None:
                colors = random_color(rgb=True).astype(np.float64)
            else:
                colors = color

            if not is_without_mask:
                image_copy = cv2.rectangle(image_copy, (bbox_f[0], bbox_f[1]), (bbox_f[2], bbox_f[3]), colors,
                                           thickness)
            else:
                mask = np.zeros_like(image_copy, np.uint8)
                mask1 = cv2.rectangle(mask, (bbox_f[0], bbox_f[1]), (bbox_f[2], bbox_f[3]), colors, -1)
                mask = np.zeros_like(image_copy, np.uint8)
                mask2 = cv2.rectangle(mask, (bbox_f[0], bbox_f[1]), (bbox_f[2], bbox_f[3]), colors, thickness)
                mask2 = cv2.addWeighted(mask1, 0.5, mask2, 8, 0.0)
                image_copy = cv2.addWeighted(image_copy, 1.0, mask2, 0.6, 0.0)
            if len(bbox) == 5 or len(bbox) == 6:
                cv2.putText(image_copy, txt, (bbox_f[0], bbox_f[1] - 2),
                            font, font_scale, (255, 255, 255), thickness=thickness, lineType=cv2.LINE_AA)
        if is_show:
            ImageHelper.show_img(image_copy, names, wait_time_ms)
        return image_copy

    @staticmethod
    def show_keypoint(image, keypoints_nx2, color=(0, 0, 255),
                      thickness=-1, radius=15, wait_time_ms=0, names=None,
                      is_show=True):
        """
        Visualize keypoint in keypoint detection bg drawing circle.

        :param image: numpy.ndarray.
        :param keypoints_nx2: list: shape is nx2, like [[0, 1], [10, 10], ...]
        :param color: tuple.
        :param thickness: int.
        :param radius: float.
        :param wait_time_ms: int
        :param names: string: display window name.
        :param is_show: bool: whether to display during middle process
        :return: numpy.ndarray.
        """
        assert image is not None
        keypoints_nx2 = np.array(keypoints_nx2, np.int64)
        image_copy = image.copy()
        for keypoint in keypoints_nx2:
            cv2.circle(image_copy, tuple(keypoint), radius=radius, color=color, thickness=thickness)
        if is_show:
            ImageHelper.show_img(image_copy, names, wait_time_ms)
        return image_copy

    @staticmethod
    def show_tensor(tensor, resize_hw=None, top_k=50, mode='CHW', is_show=True,
                    wait_time_ms=0, show_split=True, is_merge=True, row_col_num=(1, -1)):
        """

        :param wait_time_ms:
        :param tensor: torch.tensor
        :param resize_hw: list:
        :param top_k: int
        :param mode: string: 'CHW' , 'HWC'
        """

        def normalize_numpy(array):
            max_value = np.max(array)
            min_value = np.min(array)
            array = (array - min_value) / (max_value - min_value)
            return array

        assert tensor.dim() == 3, 'Dim of input tensor should be 3, please check your tensor dimension!'

        # 默认tensor格式,通道在前
        if mode == 'CHW':
            tensor = tensor
        else:
            tensor = tensor.permute(2, 0, 1)

        # 利用torch中的resize函数进行插值, 选择双线性插值平滑
        if resize_hw is not None:
            tensor = tensor[None]
            tensor = F.interpolate(tensor, resize_hw, mode='bilinear')
            tensor = tensor.squeeze(0)

        tensor = tensor.permute(1, 2, 0)

        channel = tensor.shape[2]

        if tensor.device == 'cpu':
            tensor = tensor.detach().numpy()
        else:
            tensor = tensor.cpu().detach().numpy()
        if not show_split:
            # sum可能会越界，所以需要归一化
            sum_tensor = np.sum(tensor, axis=2)
            sum_tensor = normalize_numpy(sum_tensor) * 255
            sum_tensor = sum_tensor.astype(np.uint8)

            # # mean可能值会太小,所以也要做归一化
            # mean_tensor = np.mean(tensor, axis=2)
            # mean_tensor = normalize_numpy(mean_tensor) * 255
            # mean_tensor = mean_tensor.astype(np.uint8)

            # 热力图显示
            sum_tensor = cv2.applyColorMap(np.uint8(sum_tensor), cv2.COLORMAP_JET)
            # mean_tensor = cv2.applyColorMap(np.uint8(mean_tensor), cv2.COLORMAP_JET)

            if is_show:
                ImageHelper.show_img([sum_tensor], ['sum'], wait_time_ms=wait_time_ms)

            return [sum_tensor]
        else:
            assert top_k > 0, 'top k should be positive!'
            channel_sum = np.sum(tensor, axis=(0, 1))
            index = np.argsort(channel_sum)
            select_index = index[:top_k]
            tensor = tensor[:, :, select_index]
            tensor = np.clip(tensor, 0, np.max(tensor))

            single_tensor_list = []
            if top_k > channel:
                top_k = channel
            for c in range(top_k):
                single_tensor = tensor[..., c]
                single_tensor = normalize_numpy(single_tensor) * 255
                single_tensor = single_tensor.astype(np.uint8)

                single_tensor = cv2.applyColorMap(np.uint8(single_tensor), cv2.COLORMAP_JET)
                single_tensor_list.append(single_tensor)

            if is_merge:
                return_imgs = ImageHelper.merge_imgs(single_tensor_list, row_col_num=row_col_num)
            else:
                return_imgs = single_tensor_list

            if is_show:
                ImageHelper.show_img(return_imgs, wait_time_ms=wait_time_ms, is_merge=is_merge)

            return return_imgs