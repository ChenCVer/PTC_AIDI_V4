import os
import cv2
import mmcv
import torch
import numpy as np
import torch.nn as nn
from mmcv import imshow, imwrite
from core.models.base import BaseModel
from core.models.registry import MODELS
from core.models.builder import build_backbone, build_head, build_neck
from rraitools.visualtools.feature_vistools import ClsGradClassActivationMappingVis


@MODELS.register_module
class BaseClassifier(BaseModel):

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 train_cfg=None,
                 valid_cfg=None,
                 test_cfg=None,
                 pretrained=None):

        super(BaseClassifier, self).__init__()

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
        super(BaseClassifier, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
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

    def forward_train(self, img, img_metas, label,
                      target_ignore=None, **kwargs):
        # backbone + neck
        x = self.extract_feat(img)
        # head
        losses = self.head.forward_train(x, img_metas, label,
                                         target_ignore, **kwargs)
        return losses

    def simple_test(self,
                     imgs,
                     img_metas=None,
                     **kwargs):

        # 为兼容grid-cam
        if imgs.ndim == 3:
            imgs = imgs.unsqueeze(0)

        samples_per_gpu = imgs.size(0)
        assert samples_per_gpu == 1

        x = self.extract_feat(imgs)
        outs = self.head(x)

        # skip post-processing when exporting to onnx
        if torch.onnx.is_in_onnx_export():
            return outs

        # post-process
        results = self.head.get_results(outs, img_metas, self.test_cfg)

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
                    data_batch=None,  # 用于cam
                    out_cam=False,
                    thickness=1,
                    font_scale=0.5,
                    win_name='result',
                    show=False,
                    wait_time=10,
                    out_file=None,
                    **kwargs):

        img = mmcv.imread(img)
        img = img.copy()
        result = torch.softmax(result, dim=1)
        pred_cls_id = torch.argmax(result, dim=1).detach().cpu().numpy()[0]
        pred_cls_conf = result[0][pred_cls_id].detach().cpu().numpy()

        font = cv2.FONT_HERSHEY_DUPLEX
        predict_id_text = "p_conf: " + str(np.round(pred_cls_conf, 3))
        predict_conf_text = "p_id: " + str(pred_cls_id)

        img = cv2.putText(img, predict_conf_text, (5, 20), font, font_scale, (0, 255, 0), thickness)
        img = cv2.putText(img, predict_id_text, (5, 45), font, font_scale, (0, 255, 0), thickness)

        if img_metas is not None:
            target_cls_id = img_metas[0]["label"]
            target_text = "t_id: " + str(target_cls_id)
            img = cv2.putText(img, target_text, (5, 70), font, font_scale, (0, 0, 255), thickness)

        # 保存热力图文件
        if out_cam:
            cam_img = self.get_cam(model, data_batch)
            cam_img = mmcv.imresize(cam_img, (img.shape[1], img.shape[0]))
            file_type_endswith = os.path.splitext(out_file)[-1]
            out_cam_file = out_file[:-len(file_type_endswith)] + "_cam" + file_type_endswith
            imwrite(cam_img, out_cam_file)

        if show:
            imshow(img, win_name, wait_time)

        # 保存原图+预测类别+预测概率
        if out_file is not None:
            if out_cam:
                img = cv2.addWeighted(img, 0.7, cam_img, 0.3, 0)
            imwrite(img, out_file)

    def get_cam(self, model, data):
        """
        function: 此函数用来获取热力图
        """
        grid_cam = ClsGradClassActivationMappingVis(model)
        grid_cam.set_hook_style(3, data["img"][0].shape)
        cam_img = grid_cam.run(data["img"][0])["cam_img"]

        return cam_img