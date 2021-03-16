import mmcv
import inspect
import numpy as np
from rraitools import misc
from core.core import PolygonMasks
from ...registry import PIPELINES

try:
    import albumentations
    from albumentations import Compose, BboxParams, KeypointParams
except ImportError:
    albumentations = None
    Compose = None


__all__ = [
    'Albu',
    'AlbuClsSeg',
    'AlbuKeypoint',
]


def albu_builder(cfg):
    """Import a module from albumentations.
        Inherits some of `build_from_cfg` logic.

        Args:
            cfg (dict): Config dict. It should at least contain the key "type".
        Returns:
            obj: The constructed object.
    """
    assert isinstance(cfg, dict) and 'type' in cfg
    args = cfg.copy()

    obj_type = args.pop('type')
    if misc.is_str(obj_type):
        if albumentations is None:
            raise RuntimeError('albumentations is not installed')
        obj_cls = getattr(albumentations, obj_type)
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError(
            'type must be a str or valid type, but got {}'.format(type(obj_type)))

    if 'transforms' in args:
        args['transforms'] = [
            albu_builder(transform)
            for transform in args['transforms']
        ]

    return obj_cls(**args)


@PIPELINES.register_module
class Albu(object):
    # TODO: 2020-11-08, 此类针对目标检测的album数据增强, 后续考虑和下面的AlbumClseg合并.
    """Albumentation augmentation for detection.

    Adds custom transformations from Albumentations library.
    Please, visit `https://albumentations.readthedocs.io`
    to get more information.

    An example of ``transforms`` is as followed:

    .. code-block::

        [
            dict(
                type='ShiftScaleRotate',
                shift_limit=0.0625,
                scale_limit=0.0,
                rotate_limit=0,
                interpolation=1,
                p=0.5),
            dict(
                type='RandomBrightnessContrast',
                brightness_limit=[0.1, 0.3],
                contrast_limit=[0.1, 0.3],
                p=0.2),
            dict(type='ChannelShuffle', p=0.1),
            dict(
                type='OneOf',
                transforms=[
                    dict(type='Blur', blur_limit=3, p=1.0),
                    dict(type='MedianBlur', blur_limit=3, p=1.0)
                ],
                p=0.1),
        ]

    Args:
        transforms (list[dict]): A list of albu transformations
        bbox_params (dict): Bbox_params for albumentation `Compose`
        keymap (dict): Contains {'input key':'albumentation-style key'}
        skip_img_without_anno (bool): Whether to skip the image if no ann left
            after aug
    """

    def __init__(self,
                 transforms,
                 bbox_params=None,
                 keymap=None,
                 update_pad_shape=False,
                 skip_img_without_anno=False):
        if Compose is None:
            raise RuntimeError('albumentations is not installed')

        self.transforms = transforms
        self.filter_lost_elements = False
        self.update_pad_shape = update_pad_shape
        self.skip_img_without_anno = skip_img_without_anno

        # A simple workaround to remove masks without boxes
        if (isinstance(bbox_params, dict) and 'label_fields' in bbox_params
                and 'filter_lost_elements' in bbox_params):
            self.filter_lost_elements = True
            self.origin_label_fields = bbox_params['label_fields']
            bbox_params['label_fields'] = ['idx_mapper']
            del bbox_params['filter_lost_elements']

        self.bbox_params = (
            self.albu_builder(bbox_params) if bbox_params else None)
        self.aug = Compose([self.albu_builder(t) for t in self.transforms],
                           bbox_params=self.bbox_params)

        if not keymap:
            self.keymap_to_albu = {
                'img': 'image',
                'gt_masks': 'masks',
                'gt_bboxes': 'bboxes'
            }
        else:
            self.keymap_to_albu = keymap
        self.keymap_back = {v: k for k, v in self.keymap_to_albu.items()}

    def albu_builder(self, cfg):
        """Import a module from albumentations.
        Inherits some of `build_from_cfg` logic.

        Args:
            cfg (dict): Config dict. It should at least contain the key "type".
        Returns:
            obj: The constructed object.
        """

        assert isinstance(cfg, dict) and 'type' in cfg
        args = cfg.copy()

        obj_type = args.pop('type')
        if mmcv.is_str(obj_type):
            if albumentations is None:
                raise RuntimeError('albumentations is not installed')
            obj_cls = getattr(albumentations, obj_type)
        elif inspect.isclass(obj_type):
            obj_cls = obj_type
        else:
            raise TypeError(
                f'type must be a str or valid type, but got {type(obj_type)}')

        if 'transforms' in args:
            args['transforms'] = [
                self.albu_builder(transform)
                for transform in args['transforms']
            ]

        return obj_cls(**args)

    @staticmethod
    def mapper(d, keymap):
        """
        Dictionary mapper.
        Renames keys according to keymap provided.

        Args:
            d (dict): old dict
            keymap (dict): {'old_key':'new_key'}
        Returns:
            dict: new dict.
        """

        updated_dict = {}
        for k, v in zip(d.keys(), d.values()):
            new_k = keymap.get(k, k)
            updated_dict[new_k] = d[k]
        return updated_dict

    def __call__(self, results):
        # dict to albumentations format
        results = self.mapper(results, self.keymap_to_albu)
        # TODO: add bbox_fields
        if 'bboxes' in results:
            # to list of boxes
            if isinstance(results['bboxes'], np.ndarray):
                results['bboxes'] = [x for x in results['bboxes']]
            # add pseudo-field for filtration
            if self.filter_lost_elements:
                results['idx_mapper'] = np.arange(len(results['bboxes']))

        # TODO: Support mask structure in albu
        if 'masks' in results:
            if isinstance(results['masks'], PolygonMasks):
                raise NotImplementedError(
                    'Albu only supports BitMap masks now')
            ori_masks = results['masks']
            results['masks'] = results['masks'].masks

        results = self.aug(**results)

        if 'bboxes' in results:
            if isinstance(results['bboxes'], list):
                results['bboxes'] = np.array(
                    results['bboxes'], dtype=np.float32)
            results['bboxes'] = results['bboxes'].reshape(-1, 4)

            # filter label_fields
            if self.filter_lost_elements:

                for label in self.origin_label_fields:
                    results[label] = np.array(
                        [results[label][i] for i in results['idx_mapper']])
                if 'masks' in results:
                    results['masks'] = np.array(
                        [results['masks'][i] for i in results['idx_mapper']])
                    results['masks'] = ori_masks.__class__(
                        results['masks'], results['image'].shape[0],
                        results['image'].shape[1])

                if (not len(results['idx_mapper'])
                        and self.skip_img_without_anno):
                    return None

        if 'gt_labels' in results:
            if isinstance(results['gt_labels'], list):
                results['gt_labels'] = np.array(results['gt_labels'])
            results['gt_labels'] = results['gt_labels'].astype(np.int64)

        # back to the original format
        results = self.mapper(results, self.keymap_back)

        # update final shape
        if self.update_pad_shape:
            results['pad_shape'] = results['img'].shape

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__ + f'(transforms={self.transforms})'
        return repr_str


@PIPELINES.register_module
class AlbuClsSeg(object):
    """
    Notes: 该类主要为了分割和分类调用album官方库的数据增强类
    """
    def __init__(self, transforms):
        if Compose is None:
            raise RuntimeError('albumentations is not installed')
        self.transforms = transforms
        self.aug = None

    def __call__(self, results):
        image = results.get('img', None)
        mask = results.get('gt_semantic_seg', None)
        assert image is not None
        if self.aug is None:
            if isinstance(image, list) and len(image) > 1:  # 考虑参考图
                target = {}
                for i in range(len(image) - 1):
                    target['image' + str(i)] = 'image'
                self.aug = Compose([albu_builder(t) for t in self.transforms], additional_targets=target)
            else:
                self.aug = Compose([albu_builder(t) for t in self.transforms])
        if isinstance(image, list) and len(image) > 1:
            is_mask_list = False
            img_dict = {'image': image[0].copy()}
            if mask is not None and isinstance(mask, list):
                is_mask_list = True
                img_dict['masks'] = mask.copy()
            elif mask is not None:
                is_mask_list = False
                img_dict['mask'] = mask.copy()
            for i in range(len(image) - 1):
                img_dict['image' + str(i)] = image[i + 1].copy()
            transformed = self.aug(**img_dict)
            imgs = [transformed['image']]
            for i in range(len(image) - 1):
                imgs.append(transformed['image' + str(i)])
            results['img'] = imgs
            if mask is not None:
                results['gt_semantic_seg'] = transformed['masks'] if is_mask_list else transformed['mask']
        elif isinstance(image, list):
            # 默认，一张图片最多也就一个mask
            if mask is not None:
                transformed = self.aug(image=image[0].copy(), mask=mask.copy())
                image = transformed['image']
                results['img'] = [image]
                results['gt_semantic_seg'] = transformed['mask']
            else:
                transformed = self.aug(image=image[0].copy())
                image = transformed['image']
                results['img'] = [image]
        else:
            # 默认，一张图片最多也就一个mask
            if mask is not None:
                transformed = self.aug(image=image.copy(), mask=mask.copy())
                image = transformed['image']
                results['img'] = image
                results['gt_semantic_seg'] = transformed['mask']
            else:
                transformed = self.aug(image=image.copy())
                image = transformed['image']
                results['img'] = image

        return results


@PIPELINES.register_module
class AlbuKeypoint(object):
    def __init__(self, transforms, remove_invisible=False):
        self.remove_invisible = remove_invisible
        if Compose is None:
            raise RuntimeError('albumentations is not installed')
        self.transforms = transforms
        keypoint_params = KeypointParams(format='xy',
                                         remove_invisible=remove_invisible)
        self.aug = Compose([albu_builder(t) for t in self.transforms],
                           keypoint_params=keypoint_params)

    def __call__(self, results):
        keypoint_label = results['keypoint_label']
        keypoint_index = results.get('keypoint_index', None)
        transformed = self.aug(image=results['img'], keypoints=keypoint_label.tolist())
        results['img'] = transformed['image']
        keypoints = np.array(transformed['keypoints'])
        if not self.remove_invisible:
            # 去除越界数据, 对越界的关键点进行过滤
            index = (keypoints[:, 0] > 0) & (keypoints[:, 1] > 0) & \
                    (keypoints[:, 0] < transformed['image'].shape[1]) & \
                    (keypoints[:, 1] < transformed['image'].shape[0])
            keypoints = keypoints[index]
            if keypoint_index is not None:
                keypoint_index = keypoint_index[index]

        results['keypoint_label'] = keypoints
        if keypoint_index is not None:
            results['keypoint_index'] = keypoint_index
        return results