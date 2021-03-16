import math
import torch
import numpy as np
import torch.nn as nn
from .brick import xyxy2xywh
from mmcv.cnn import normal_init
from core.core import multi_apply
from core.core import multiclass_nms
from core.models.registry import HEADS
from core.models.builder import build_loss
from .dense_test_mixins import BBoxTestMixin
from core.models.components.heads.base import BaseHead


@HEADS.register_module()
class Yolov5Head(BaseHead, BBoxTestMixin):

    def __init__(self,
                 depth_multiple,
                 width_multiple,
                 label_smooth=True,
                 conf_balances=[0.4, 1, 4],
                 deta=0.01,
                 anchors=[[142, 110], [192, 243], [459, 401],
                          [36, 75], [76, 55], [72, 146],
                          [12, 16], [19, 36], [40, 28]],
                 anchors_mask=[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                 strides=[32, 16, 8],
                 bbox_weight=False,
                 in_channels=[128, 256, 512],
                 num_classes=80,
                 nms_type='nms',
                 nms_thr=.5,
                 ignore_thre=.3,
                 anchor_t=4,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0,
                     reduction='sum'),
                 loss_conf=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0,
                     reduction='sum'),
                 loss_bbox=dict(
                     type='GIoULoss',
                     loss_weight=10.0),
                 train_cfg=None,
                 valid_cfg=None,
                 test_cfg=None):

        super(Yolov5Head, self).__init__()

        self.depth_multiple = depth_multiple
        self.width_multiple = width_multiple

        self.train_cfg = train_cfg
        self.valid_cfg = valid_cfg
        self.test_cfg = test_cfg

        self.conf_balances = conf_balances
        self.num_classes = num_classes
        self.nms_thr = nms_thr
        self.nms_type = nms_type
        self.out_channels = []
        self.in_channels = in_channels
        self.base_num = 5
        self.bbox_weight = bbox_weight
        self.anchor_t = anchor_t
        for mask in anchors_mask:
            self.out_channels.append(len(mask) * (self.base_num + num_classes))

        self.anchors = anchors
        self.anchor_masks = anchors_mask
        self.down_ratios = strides
        if label_smooth is None or not isinstance(label_smooth, bool):
            label_smooth = False
        self.label_smooth = label_smooth
        self.deta = deta
        self.ignore_thre = ignore_thre
        self.bbox_loss = build_loss(loss_bbox)
        self.conf_loss = build_loss(loss_conf)
        self.cls_loss = build_loss(loss_cls)

        self._init_layers()

    @property
    def num_levels(self):
        return len(self.down_ratios)

    @property
    def num_attrib(self):
        """int: number of attributes in pred_map, bboxes (4) +
        objectness (1) + num_classes"""

        return 5 + self.num_classes

    def _init_layers(self):
        self.convs_pred = nn.ModuleList()
        output = len(self.anchor_masks) * (self.base_num + self.num_classes)
        for idx, in_channels in enumerate(self.in_channels):
            divisible = math.ceil(in_channels * self.width_multiple / 8) * 8
            conv_pred = nn.Conv2d(divisible, output, kernel_size=1)
            self.convs_pred.append(conv_pred)

    def init_weights(self):
        """Initialize weights of the head: nn.Conv2d"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple[Tensor]: A tuple of multi-level predication map, each is a
                4D-tensor of shape (batch_size, 5+num_classes, height, width).
        """
        assert len(feats) == self.num_levels
        pred_maps = []
        for idx in range(self.num_levels):
            x = feats[idx]
            pred_map = self.convs_pred[idx](x)
            pred_maps.append(pred_map)

        return tuple(pred_maps),

    def get_results(self, pred_maps, img_metas,
                   cfg=None, rescale=False, with_nms=True):
        """Transform network output for a batch into bbox predictions.

        Args:
            pred_maps (list[Tensor]): Raw predictions for a batch of images.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used
            rescale (bool): If True, return boxes in original image space

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of the
                corresponding box.
        """
        result_list = []
        num_levels = len(pred_maps)
        for img_id in range(len(img_metas)):
            pred_maps_list = [pred_maps[i][img_id].detach() for i in range(num_levels)]
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self._get_bboxes_single(pred_maps_list, scale_factor,
                                                cfg, rescale, with_nms)
            result_list.append(proposals)

        return result_list

    def _get_bboxes_single(self,
                           pred_maps_list,
                           scale_factor,
                           cfg,
                           rescale=False,
                           with_nms=True):
        """
        Transform outputs for a single batch item into bbox predictions.

        Args:
            pred_maps_list (list[Tensor]): Prediction maps for different scales
                of each single image in the batch.
            scale_factor (ndarray): Scale factor of the image arrange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.

        Returns:
            Tensor: Labeled boxes in shape (n, 5), where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(pred_maps_list) == self.num_levels
        multi_lvl_bboxes = []
        multi_lvl_cls_scores = []
        multi_lvl_conf_scores = []

        # 循环处理每一层.
        for i, mask in enumerate(self.anchor_masks):
            _, ny, nx = pred_maps_list[i].shape  # x(255,20,20) to x(num_masks,h,w,num_attr)
            pred_maps_list[i] = pred_maps_list[i].view(len(mask), self.num_attrib,
                                                       ny, nx).permute(0, 2, 3, 1).contiguous()
            yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
            grid_xy = torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float().to(pred_maps_list[i].device)
            y = pred_maps_list[i].sigmoid()  # 对整体进行sigmoid操作.
            # xy 映射回原图位置
            y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid_xy) * self.down_ratios[i]
            # 映射回原图wh尺寸
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * torch.from_numpy(
                np.array([self.anchors[idx] for idx in mask])).to(
                y.device).view(1, -1, 1, 1, 2)
            # y.shape -> (-1, num_attr)
            y = y.view(-1, self.num_attrib)
            # bbox, conf and cls
            bbox_pred = y[..., :4]
            conf_pred = y[..., 4].view(-1)
            cls_pred = y[..., 5:].view(-1, self.num_classes)  # Cls pred one-hot.

            # Filtering out all predictions with conf < conf_thr
            conf_thr = cfg.get('conf_thr', -1)
            conf_inds = conf_pred.ge(conf_thr).nonzero().flatten()
            bbox_pred = bbox_pred[conf_inds, :]
            cls_pred = cls_pred[conf_inds, :]
            conf_pred = conf_pred[conf_inds]

            # Get top-k prediction
            nms_pre = cfg.get('nms_pre', -1)
            if 0 < nms_pre < conf_pred.size(0):
                _, topk_inds = conf_pred.topk(nms_pre)
                bbox_pred = bbox_pred[topk_inds, :]
                cls_pred = cls_pred[topk_inds, :]
                conf_pred = conf_pred[topk_inds]

            # Save the result of current scale
            multi_lvl_bboxes.append(bbox_pred)
            multi_lvl_cls_scores.append(cls_pred)
            multi_lvl_conf_scores.append(conf_pred)

        # Merge the results of different scales together
        multi_lvl_bboxes = torch.cat(multi_lvl_bboxes)
        multi_lvl_cls_scores = torch.cat(multi_lvl_cls_scores)
        multi_lvl_conf_scores = torch.cat(multi_lvl_conf_scores)

        if with_nms and multi_lvl_conf_scores.size(0) == 0:
            return torch.zeros((0, 5)), torch.zeros((0,))

        if rescale:
            multi_lvl_bboxes /= multi_lvl_bboxes.new_tensor(scale_factor)

        # In mmdet 2.x, the class_id for background is num_classes.
        # i.e., the last column.
        padding = multi_lvl_cls_scores.new_zeros(multi_lvl_cls_scores.shape[0], 1)
        multi_lvl_cls_scores = torch.cat([multi_lvl_cls_scores, padding], dim=1)

        if with_nms:
            det_bboxes, det_labels = multiclass_nms(
                multi_lvl_bboxes,
                multi_lvl_cls_scores,
                cfg.score_thr,
                cfg.nms,
                cfg.max_per_img,
                score_factors=multi_lvl_conf_scores)
            return det_bboxes, det_labels
        else:
            return (multi_lvl_bboxes, multi_lvl_cls_scores,
                    multi_lvl_conf_scores)

    def get_targets(self,
                    pred,
                    img_metas,
                    gt_bboxes,
                    gt_labels):

        device = pred[0].device
        # todo: 这里需要将gt_bbox兼容, 需要将gt_bbox首先做归一化处理.
        for idx, gt_bbox in enumerate(gt_bboxes):
            gt_bboxes[idx][:, [1, 3]] /= img_metas[idx]["pad_shape"][0]  # y/h
            gt_bboxes[idx][:, [0, 2]] /= img_metas[idx]["pad_shape"][1]  # x/w

        gain = torch.ones(6, device=device)  # normalized to gridspace gain
        ft = torch.cuda.FloatTensor if pred[0].is_cuda else torch.Tensor
        targets = ft([]).to(device)

        for i, gt_bbox in enumerate(gt_bboxes):
            gt_label = gt_labels[i].float()
            img_idx = torch.ones(len(gt_bbox), 1, device=device) * i
            targets = torch.cat((targets, torch.cat((img_idx, gt_label[:, None], gt_bbox), dim=-1)))

        na, nt = len(self.anchor_masks), len(targets)
        tcls, tbox, indices, anch, ignore_mask = [], [], [], [], []
        targets[..., 2:] = xyxy2xywh(targets[..., 2:])
        g = 0.5  # offset grid中心偏移
        # overlap offsets 按grid区域换算偏移区域,附近的4个网格上下左右
        off = torch.tensor([[1, 0], [0, 1],
                            [-1, 0], [0, -1]],
                           device=device).float()
        # anchor tensor, same as .repeat_interleave(nt)
        at = torch.arange(na).view(na, 1).repeat(1, nt)
        for idx, (mask, down_ratio) in enumerate(zip(self.anchor_masks, self.down_ratios)):

            anchors = np.array(self.anchors, dtype=np.float32)[mask] / down_ratio  # Scale
            # for i in range(len(self.anchor_masks)):
            #     anchors = self.anchors[i]
            gain[2:] = torch.tensor(pred[idx].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            a, t, offsets = [], targets * gain, 0
            if nt:
                r = t[None, :, 4:6] / torch.from_numpy(anchors[:, None]).to(device)  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.anchor_t  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']
                # iou(3,n) = wh_iou(anchors(3,2), gwh(n,2))
                a, t = at[j], t.repeat(na, 1, 1)[j]

                # overlaps
                gxy = t[:, 2:4]  # grid xy
                z = torch.zeros_like(gxy)
                # j,k 为小于0.5的偏移 ,l,m为大于0.5的偏移
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxy % 1. > (1 - g)) & (gxy < (gain[[2, 3]] - 1.))).T
                # t 原始target, t[j] x<.5 偏移的target, t[k] y<.5 偏移的target,
                # t[l] x>.5 偏移的target, t[m] y>.5 偏移的target
                a, t = torch.cat((a, a[j], a[k], a[l], a[m]), 0), \
                       torch.cat((t, t[j], t[k], t[l], t[m]), 0)
                # z 原始target,x<0.5 +0.5 ,y<0.5 +0.5,x>.5 -0.5,y>.5 -0.5
                offsets = torch.cat((z, z[j] + off[0], z[k] + off[1],
                                     z[l] + off[2], z[m] + off[3]), 0) * g

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()  # 获取所有的grid 位置 -0.5<offsets<0.5
            gi, gj = gij.T  # grid xy indices

            # Append
            indices.append((b, a, gj, gi))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box x,y 偏移范围在[-0.5,1.5]
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class
            ignore_mask.append([])

        return indices, tbox, tcls, anch, ignore_mask

    def get_pred(self, preds):
        preds_result = []
        for index, mask in enumerate(self.anchor_masks):
            pred = preds[index]
            batch = pred.shape[0]
            grid_h, grid_w = pred.shape[-2:]
            out_ch = self.base_num + self.num_classes
            # [nB, nA, nC, nH, nW]
            pred = pred.view(batch, len(mask), out_ch, grid_h, grid_w)
            # 维度变化: [nB, nA, nC, nH, nW] -> [nB, nA, nH, nW, nC]
            pred = pred.permute(0, 1, 3, 4, 2)
            preds_result.append(pred)
        return preds_result

    def loss(self,
             input,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        # outs + (gt_bboxes, gt_labels, img_metas)
        pred = self.get_pred(input)
        indices, tbox, tcls, \
        ancher, ignore_mask = self.get_targets(pred=pred,
                                               img_metas=img_metas,
                                               gt_bboxes=gt_bboxes,  # gt_bbox为原图尺度.
                                               gt_labels=gt_labels)

        bbox_loss, conf_loss, cls_loss = multi_apply(self.loss_single, pred,
                                                     indices, tbox, tcls, ancher,
                                                     self.conf_balances, ignore_mask)

        return dict(bbox_loss=bbox_loss, conf_loss=conf_loss, cls_loss=cls_loss)

    def loss_single(self,
                    pred,
                    indices,
                    tbox,
                    tcls,
                    anchors,
                    conf_balances,
                    ignore_mask):

        device = pred.device
        ft = torch.cuda.FloatTensor if pred.is_cuda else torch.Tensor
        lcls, lbox, lobj = ft([0]).to(device), ft([0]).to(device), ft([0]).to(device)

        # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
        def smooth_BCE(eps=0.1):
            # return positive, negative label smoothing BCE targets
            return 1.0 - 0.5 * eps, 0.5 * eps

        # class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        eps = 0
        if self.label_smooth and self.deta > 0:
            eps = self.deta
        cp, cn = smooth_BCE(eps=eps)

        # per output
        nt = 0  # number of targets
        pi = pred
        b, a, gj, gi = indices  # image, anchor, gridy, gridx
        tobj = torch.zeros_like(pi[..., 0]).to(device)  # target obj

        nb = b.shape[0]  # number of targets
        if nb:
            nt += nb  # cumulative targets
            ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

            # iou loss: GIoU
            pxy = ps[:, :2].sigmoid() * 2. - 0.5  # -0.5<pxy<1.5
            pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * torch.from_numpy(anchors).to(
                device)  # 0-4倍缩放 model.hyp['anchor_t']=4
            pbox = torch.cat((pxy, pwh), 1).to(device)  # predicted box
            bbox_loss, giou = self.bbox_loss(pbox, tbox)
            lbox += bbox_loss

            # conf loss: contain giou ratio
            # tobj[b, a, gj, gi] = (1.0 - model.gr) +
            # model.gr * giou.detach().clamp(0).type(tobj.dtype)
            tobj[b, a, gj, gi] = giou.detach().clamp(0).type(tobj.dtype)  # giou ratio

            # cls loss
            t = torch.full_like(ps[:, self.base_num:], cn).to(device)  # targets
            # t[range(nb), tcls] = cp
            t[range(nb), tcls.squeeze()] = cp
            # TODO: 作者源代码lcls += self.cls_loss(ps[:, self.base_num:], t, self.num_classes)
            lcls += self.cls_loss(ps[:, self.base_num:], t)

        lobj += self.conf_loss(pi[..., 4], tobj) * conf_balances  # obj loss

        return lbox, lobj, lcls
