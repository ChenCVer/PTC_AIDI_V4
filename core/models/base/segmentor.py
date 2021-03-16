import cv2
import mmcv
import torch
import numpy as np
import torch.nn as nn
from mmcv import imshow, imwrite
from core.models.base import BaseModel
from core.models.registry import MODELS
from core.models.builder import build_backbone, build_head, build_neck
from core.models.components.utils import slide_window_crop_tensor, overlap_crop_inv_max

__all__ = ["BaseSegmentor", ]


@MODELS.register_module
class BaseSegmentor(BaseModel):
    """Base class for segmentation.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 train_cfg=None,
                 valid_cfg=None,
                 test_cfg=None,
                 pretrained=None):

        super(BaseSegmentor, self).__init__()

        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        head.update(train_cfg=train_cfg)
        head.update(valid_cfg=valid_cfg)
        head.update(test_cfg=test_cfg)
        self.head = build_head(head)

        self.train_cfg = train_cfg
        self.valid_cfg = valid_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        # init backbone
        super(BaseSegmentor, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        # init neck
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        # init head
        if self.with_head:
            self.head.init_weights()

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck
        """
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_flops(self, img):
        """Used for computing network flops.
        """
        x = self.extract_feat(img)
        outs = self.head(x)
        return outs

    def forward_train(self, img, img_metas, gt_semantic_seg,
                      target_ignore=None, **kwargs):
        # backbone + neck
        x = self.extract_feat(img)
        # head
        losses = self.head.forward_train(x, img_metas, gt_semantic_seg,
                                         target_ignore, **kwargs)
        return losses

    def simple_test(self,
                    imgs,
                    img_metas,
                    rescale=True,
                    **kwargs):
        """Test function without test time augmentation
        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            np.ndarray: cnts
        """
        # select test mode
        test_mode = self.test_cfg.get('type')
        # test in piece mode
        if test_mode == "piece_img_mode":
            piece_shape = self.test_cfg.get("piece_shape", None)
            assert piece_shape is not None, 'piece_shape must be not None!'
            overlap_hw = self.test_cfg.get("overlap_hw", (0, 0))
            num_classes = self.test_cfg.get("num_classes")
            piece_tensor_list, pts_list, \
            maps = slide_window_crop_tensor(imgs, piece_shape, overlap_hw)
            outs = []
            for idx, piece_tensor in enumerate(piece_tensor_list):
                x = self.extract_feat(piece_tensor)
                outs.append(self.head(x))
            img_shape = (1, num_classes, imgs.shape[2], imgs.shape[3])
            outs = overlap_crop_inv_max(outs, pts_list, maps, img_shape)
        # test in whole mode
        else:
            x = self.extract_feat(imgs)
            outs = self.head(x)

        # skip post-processing when exporting to onnx
        if torch.onnx.is_in_onnx_export():
            return outs

        # post-process
        results = self.head.get_results(outs, img_metas,
                                        self.test_cfg, rescale=rescale)

        return results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation"""
        raise NotImplementedError

    def show_result(self,
                    model,
                    img,
                    result,
                    img_metas=None,
                    cfg=None,
                    show_cnts=True,
                    win_name='result',
                    show=False,
                    wait_time=0,
                    out_file=None,
                    **kwargs):

        img = mmcv.imread(img)
        img = img.copy()

        # filter
        # put cnts into orig img, meanwhile maybe filter cnts which are maybe useless.
        if show_cnts:
            mask = np.zeros_like(result)
            predictive = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            score_thr = self.test_cfg.get('score_thr', None)
            if score_thr is None:
                _, predictive = cv2.threshold(predictive, 0, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(predictive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Here you maybe need to count the area of each bbox and use it to filter.
            cnt_count = 0
            area_thr = self.test_cfg.get('area_thr', 0.)
            for idx, cnt in enumerate(contours):
                # solve the min area rect for cnt
                min_rect = cv2.minAreaRect(cnt)
                center = min_rect[0]
                area = min_rect[1][0] * min_rect[1][1]
                box = np.int0(cv2.boxPoints(min_rect))
                if area < area_thr:
                    cv2.drawContours(mask, [box], 0, (0, 0, 255), 2)
                    # cv2.circle(mask, (int(center[0]), int(center[1])), 2, (0, 255, 0), -1)
                else:
                    cv2.drawContours(mask, [box], 0, (0, 255, 0), 2)
                    # cv2.circle(mask, (int(center[0]), int(center[1])), 2, (0, 255, 0), -1)
                cnt_count += 1

            cv2.putText(mask, str(cnt_count), (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

            result = mask

        final_result = cv2.addWeighted(img, 0.7, result, 0.3, 0)

        if show:
            imshow(final_result, win_name, wait_time)

        if out_file is not None:
            imwrite(final_result, out_file)

        return final_result
