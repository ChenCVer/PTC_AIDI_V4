import math
import torch
import torch.nn as nn
from .brick import xywh2xyxy
import torch.nn.functional as F
from mmcv.cnn import ConvModule, normal_init
from core.core import multi_apply, multiclass_nms
from core.models.registry import HEADS
from core.models.builder import build_loss
from .dense_test_mixins import BBoxTestMixin
from core.models.components.heads.base import BaseHead
from core.core import bbox_ious

__all__ = ["YoloHead",
           ]


@HEADS.register_module()
class YoloHead(BaseHead, BBoxTestMixin):
    """YOLOV3Head Paper link: https://arxiv.org/abs/1804.02767.

    Args:
        num_classes (int): The number of object classes (w/o background)
        in_channels (List[int]): Number of input channels per scale.
        out_channels (List[int]): The number of output channels per scale
            before the final 1x1 layer. Default: (1024, 512, 256).
            Should be in descending order. Default: (32, 16, 8).
        one_hot_smoother (float): Set a non-zero value to enable label-smooth
            Default: 0.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN', requires_grad=True)
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='LeakyReLU', negative_slope=0.1).
        loss_cls (dict): Config of classification loss.
        loss_conf (dict): Config of confidence loss.
        loss_bbox (dict): Config of xy coordinate loss.
        train_cfg (dict): Training config of YOLOV3 head. Default: None.
        test_cfg (dict): Testing config of YOLOV3 head. Default: None.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 out_channels=(1024, 512, 256),
                 anchors=[[116, 90], [156, 198], [373, 326],
                          [30, 61], [62, 45], [59, 119],
                          [10, 13], [16, 30], [33, 23]],
                 anchors_mask=[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                 strides=[32, 16, 8],
                 one_hot_smoother=0.0,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 noobject_scale=1.0,
                 object_scale=5.0,
                 loss_conf=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_bbox_type="iou",
                 loss_bbox=dict(
                     type='GIoULoss',
                     loss_weight=10.0),
                 loss_level_weights=[1.0, 1.0, 1.0],
                 train_cfg=None,
                 valid_cfg=None,
                 test_cfg=None):

        super(YoloHead, self).__init__()
        # Check params
        assert (len(in_channels) == len(out_channels) == len(strides))

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.anchors = torch.Tensor(anchors)
        self.strides = strides
        self.num_anchors = len(anchors_mask[0])
        self.anchors_mask = anchors_mask

        # weight of conf loss for oobject and noobject
        self.noobject_scale = noobject_scale
        self.object_scale = object_scale

        self.train_cfg = train_cfg
        self.valid_cfg = valid_cfg
        self.test_cfg = test_cfg

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        # only work when num_classes more than one.
        self.one_hot_smoother = one_hot_smoother

        self.loss_level_weights = loss_level_weights
        self.loss_cls = build_loss(loss_cls)
        self.loss_conf = build_loss(loss_conf)
        self.loss_bbox_type = loss_bbox_type

        if self.loss_bbox_type == "iou":
            assert loss_bbox.get('type').lower().find(self.loss_bbox_type) > -1, \
                   "if {0} is {1}, the bbox loss certainly is one of IOU/GIOU/DIOU" \
                   "/CIOU.".format(self.loss_bbox_type, "iou")
        self.loss_bbox = build_loss(loss_bbox)

        self._init_layers()

    @property
    def num_levels(self):
        """
        number of predmaps.
        """
        return len(self.strides)

    @property
    def num_attrib(self):
        """int: number of attributes in pred_map, bboxes (4) +
        objectness (1) + num_classes"""

        return 5 + self.num_classes

    def _init_layers(self):
        self.convs_bridge = nn.ModuleList()
        self.convs_pred = nn.ModuleList()
        for i in range(self.num_levels):
            # conv_3x3: CBL
            conv_bridge = ConvModule(
                self.in_channels[i],
                self.out_channels[i],
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
            # conv_1x1
            conv_pred = nn.Conv2d(self.out_channels[i],
                                  self.num_anchors * self.num_attrib, 1)

            self.convs_bridge.append(conv_bridge)
            self.convs_pred.append(conv_pred)

    def init_weights(self):
        """Initialize weights of the head.只需要初始化conv_pred, 而conv_bridge不需要初始化."""
        for m in self.convs_pred:
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
        for i in range(self.num_levels):
            x = feats[i]
            x = self.convs_bridge[i](x)
            pred_map = self.convs_pred[i](x)
            pred_maps.append(pred_map)

        return tuple(pred_maps),

    def loss(self, pred_maps, gt_bboxes, gt_labels,
             img_metas, gt_bboxes_ignore=None):
        """Compute loss of the head.
        Args:
            pred_maps (list[Tensor]): Prediction map for each scale level,
                shape (N, num_anchors * num_attrib, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        num_levels = len(pred_maps)
        gt_bboxes_list = [gt_bboxes, ] * num_levels
        gt_labels_list = [gt_labels, ] * num_levels
        anchors = [self.anchors] * num_levels

        losses_cls, losses_conf, losses_bbox = multi_apply(
            self.loss_single, pred_maps, gt_bboxes_list,
            gt_labels_list, anchors, self.strides,
            self.anchors_mask, self.loss_level_weights)

        return dict(
            loss_cls=losses_cls,
            loss_conf=losses_conf,
            loss_bbox=losses_bbox)

    def loss_single(self, pred_map, gt_bboxes, gt_labels,
                    anchors, feature_stride, anchors_mask,
                    loss_level_weight):
        """Compute loss of a single image from a batch.

        Args:
            pred_map (Tensor): Raw predictions for a single level.
            gt_bboxes(Tensor): The Ground-Truth bboxes for a single level.
            gt_labels(Tensor): The Ground-Truth labels for a single level.
            anchors(Tensor): the anchors for a single level.
            feature_stride(int): downsample times for feature map.
            anchors_mask(list): the same as anchors.
            loss_level_weight(float): the weight of loss for a single level.

        Returns:
            tuple:
                loss_cls (Tensor): Classification loss.
                loss_conf (Tensor): Confidence loss.
                loss_bbox (Tensor): Regression loss of bbox coordinate.
        """
        # 模型输出 (B,anchor*(x y w h conf+cls),h,w)
        nB = pred_map.size(0)  # batchsize
        nA = self.num_anchors  # anchor_nums
        nC = self.num_classes  # num_classes
        nH = pred_map.size(2)  # output_h
        nW = pred_map.size(3)  # output_w
        device = pred_map.device
        anchors = anchors.to(device) / float(feature_stride)

        # (nB, nA*[1+1+num_cls], H, W) -> (nB, nA, [1+1+num_cls], H*W)
        output = pred_map.view(nB, nA, -1, nH * nW)
        coord = torch.zeros_like(output[:, :, :4])  # x y w h
        # 这里手动进行了sigmoid操作, 则后续调用的损失函数只能是bce而不能是bce_with_logits,
        # 为了兼容, 这里暂时不进行sigmoid操作
        coord[:, :, :2] = output[:, :, :2]  # tx,ty
        coord[:, :, 2:4] = output[:, :, 2:4]  # tw,th
        conf = output[:, :, 4]  # conf

        # 对wh进行约束,防止后面exp溢出
        coord[:, :, 2:4] = torch.where(coord[:, :, 2:4] > 1e5,
                                       torch.ones_like(coord[:, :, 2:4]) * 1e5,
                                       coord[:, :, 2:4])

        if nC > 1:
            # contiguous一般用于view和transpose之前，防止浅拷贝，实际上好像是固定用法
            cls = output[:, :, 5:].contiguous().view(nB * nA, nC, nH * nW).transpose(1, 2).contiguous().view(-1, nC)

        # step1: create grid
        pred_boxes = torch.zeros(nB * nA * nH * nW, 4, dtype=torch.float, device=device)
        lin_x = torch.linspace(0, nW - 1, nW).to(device).repeat(nH, 1).view(nH * nW)
        lin_y = torch.linspace(0, nH - 1, nH).to(device).repeat(nW, 1).t().contiguous().view(nH * nW)
        anchor_w = anchors[anchors_mask, 0].view(nA, 1).to(device)
        anchor_h = anchors[anchors_mask, 1].view(nA, 1).to(device)
        # step2: decode, 注意这里pred_boxes为coord的detach()操作. 也即对pred_boxes进行各种操作
        # 不会影响coord, pred_boxes不会产生梯度. 相当于pred_boxes从coord中剥离出来成为一个新叶子
        # 节点, 即使该叶子节点有梯度, 也会在传播值coord时候被切断, 对节点的任何操作都不会影响coord.
        # 预测xy,是相对grid的左上角偏移, pred_boxes主要用于计算mask, 所以这里用detach().
        pred_boxes[:, 0] = (coord[:, :, 0].detach().sigmoid() + lin_x).view(-1)
        pred_boxes[:, 1] = (coord[:, :, 1].detach().sigmoid() + lin_y).view(-1)
        # 预测wh,是相对anchor的缩放比例,带exp
        pred_boxes[:, 2] = (coord[:, :, 2].detach().exp() * anchor_w).view(-1)
        pred_boxes[:, 3] = (coord[:, :, 3].detach().exp() * anchor_h).view(-1)

        # step3: get_targets.
        coord_mask, conf_pos_mask, conf_neg_mask, \
        cls_mask, tcoord, tconf, tcls, coord_iou = self._get_targets(pred_boxes, gt_bboxes,
                                                                     gt_labels, feature_stride,
                                                                     anchors, anchors_mask,
                                                                     nH, nW, nC)
        if nC > 1:
            tcls = tcls[cls_mask.bool()].float()  # cls_mask里面存储着正样本的mask信息
            cls_mask = cls_mask.view(-1, 1).repeat(1, nC)
            cls = cls[cls_mask.bool()].view(-1, nC)  # 提取出正样本的所有cls的概率值

        # step4: compute loss
        # regression loss
        # 这里对于回归损失, 这里考虑是否运用iou loss还是bce(xy)+mse(wh)
        if self.loss_bbox_type == 'iou':
            # iou loss
            # 由于需要梯度反传, 也即对pred_boxes的操作产生的梯度需要传送至coord中.因此, 这里需要重新进行解码操作.
            # pred_boxes也是特征图尺度.
            pred_boxes[:, 0] = (coord[:, :, 0].sigmoid() + lin_x).view(-1)  # 预测xy,是相对grid的左上角偏移
            pred_boxes[:, 1] = (coord[:, :, 1].sigmoid() + lin_y).view(-1)
            pred_boxes[:, 2] = (coord[:, :, 2].exp() * anchor_w).view(-1)  # 预测wh,是相对anchor的缩放比例,带exp
            pred_boxes[:, 3] = (coord[:, :, 3].exp() * anchor_h).view(-1)
            # reshape coord_iou: [nB, nA, 4, nH*nW] -> [nB, nA, nH*nW, 4] -> [nB*nA*nH*nW, 4]
            coord_iou = coord_iou.transpose(3, 2).reshape(-1, 4)
            coord_mask = coord_mask.expand_as(tcoord).transpose(3, 2).reshape(-1, 4)
            # xywh -> xyxy
            pred_boxes = xywh2xyxy(pred_boxes)
            coord_iou = xywh2xyxy(coord_iou)
            loss_coord = loss_level_weight * self.loss_bbox(pred_boxes, coord_iou, weight=coord_mask)
        # l1 loss
        else:
            coord_mask = coord_mask.expand_as(tcoord)
            loss_coord = loss_level_weight * self.loss_bbox(coord, tcoord, weight=coord_mask)

        # conf loss
        # 有物体的位置计算conf(置信度损失), 非常少, 有几个gt, 那么就那几个位置有loss
        loss_conf_pos = 1.0 * self.object_scale * self.loss_conf(conf, tconf, weight=conf_pos_mask)
        # 没有物体的位置计算conf,比较多，注意需要排除几种特殊情况包括：
        # (1) iou大于预测0.6位置
        # (2) 被忽略的gt
        # (3) 已经被anchor匹配上的位置
        # 要排除掉的原因是因为(1)类情况比较特殊，实际上可以认为是有物体，虽然他没有匹配gt
        loss_conf_neg = 1.0 * self.noobject_scale * self.loss_conf(conf, tconf, weight=conf_neg_mask)
        loss_conf = loss_level_weight * (loss_conf_pos + loss_conf_neg)  # 置信度损失

        # cls loss
        if nC > 1 and cls.numel() > 0:
            loss_cls = loss_level_weight * self.loss_cls(cls, tcls)
        else:
            loss_cls = torch.tensor(0.0, device=device)

        return loss_cls, loss_conf, loss_coord

    def _get_targets(self, pred_bboxes, gt_bboxes, gt_labels,
                     feature_stride, anchors, anchors_mask,
                     nH, nW, nC):
        """
        # ------------------------------darknet yolov3匹配规则说明-------------------------
        这里说下darknet中, forward_yolo_layer.c的代码的具体步骤, 然后说下本分代码的对应实现方案：
        step 1: 遍历每张图中的每个anchor, 对pred进行操作, x和y进行sigmoid()操作, 对conf和classes
                也进行sigmoid()操作;
        step 2: 遍历每张图中的每个cell位置的每个anchor, 对pred中的bbox部分进行解码操作(基于anchor
                的wh和所在cell位置)得到真正的pred_bbox. 接着遍历该张图中的所有gt,计算pred_bbox与
                每个gt的iou值, 这样就基于max_iou原则得到了pre_bbox的最佳gt(也即gt与pred_bbox的
                IOU最大), 如果其IOU大于ignore_thresh, 则pred_bbox对应的anchor记做忽略样本, 如果
                其IOU小于ignore_thresh, 暂时作为负样本看待.
        step 3: 遍历每张图片中的每一个gt, 计算该gt所在cell位置(i, j), 遍历该cell中的所有anchor,
                基于max_IOU原则, 得到gt匹配的最佳anchor. 如果gt匹配到的anchor是当前层的anchor,
                则该anchor记为正样本.
        step 4: 在step3的基础上, 遍历当前层cell(i, j)中的anchor中除了最佳的anchor之外的其他几个
                anchor, 计算gt这些anchor的iou值, 如果其iou值大于设定值:l.iou_thresh, 则该anchor
                也负责预测该gt, 也即记做正样本.
        # ------------------------------------------------------------------------------
        Compare prediction boxes and ground truths, convert ground
        truths to network output tensors """
        # Parameters
        # pred_boxes=nB * nA * nH * nW, 4
        # ground_truth=nB个list
        ground_truth = gt_bboxes
        nB = len(ground_truth)
        nA = self.num_anchors
        nAnchors = nA * nH * nW
        device = pred_bboxes.device

        # Tensors, nA行, nH×nW列
        # 正样本掩码矩阵.
        conf_pos_mask = torch.zeros(nB, nA, nH * nW, requires_grad=False, device=device)
        # 1_maxIOU<Thresh, 默认所有样本首先是负样本, conf_neg_mask中为0表示忽略样本.
        conf_neg_mask = torch.ones(nB, nA, nH * nW, requires_grad=False, device=device)
        # 坐标掩码矩阵
        coord_mask = torch.zeros(nB, nA, 1, nH * nW, requires_grad=False, device=device)
        # 类别掩码矩阵
        cls_mask = torch.zeros(nB, nA, nH * nW, requires_grad=False, dtype=torch.uint8, device=device)
        # coord target
        tcoord = torch.zeros(nB, nA, 4, nH * nW, requires_grad=False, device=device)
        # conf target
        tconf = torch.zeros(nB, nA, nH * nW, requires_grad=False, device=device)
        # iou target
        coord_iou = torch.zeros(nB, nA, 4, nH * nW, requires_grad=False, device=device)  # iou loss
        # cls target
        if nC > 1:
            tcls = torch.zeros(nB, nA, nH * nW, nC, requires_grad=False, device=device)
        else:
            tcls = torch.zeros(nB, nA, nH * nW, requires_grad=False, device=device)

        all_gts = 0       # all gts of images in a batchsize.
        all_pos_nums = 0  # all pos samples in a some one predmap.

        for b in range(nB):  # 遍历每一张图片
            gt_bboxs = ground_truth[b]  # 获取图片中的所有gt_bboxes: xyxy
            gt_label = gt_labels[b]  # 获取图片中所有gt对应的类别id
            if len(gt_bboxs) == 0:  # No gt for this image
                continue
            all_gts += len(gt_bboxs)  # 该图片中gt的个数
            # Build up tensors
            # (nAnchors,4),已经还原到特征图尺度了, 获取该图片对应的全部预测bbox
            # cur_pred_boxes这里是指获取某一张图片所有的pred_bbox(已解码)
            cur_pred_boxes = pred_bboxes[b * nAnchors:(b + 1) * nAnchors]
            anchors_wh = torch.cat([torch.zeros_like(anchors), anchors], 1)

            # gt在每个预测层, 首先都是原图尺度输入, 输入的是xyxy->需要转化为xywh格式计算, 然后降到
            # 特定的下采样倍数.
            gt = torch.zeros(len(gt_bboxs), 4, device=device)  # 特征图尺度的x,y,w,h
            for idx, bbox in enumerate(gt_bboxs):  # 遍历每一个gt
                gt[idx, 0] = ((bbox[0] + bbox[2]) / 2) / float(feature_stride)  # 降到特征图尺度
                gt[idx, 1] = ((bbox[1] + bbox[3]) / 2) / float(feature_stride)
                bbox_w = bbox[2] - bbox[0]
                bbox_h = bbox[3] - bbox[1]
                if bbox_w < 1:
                    bbox_w = 1
                if bbox_h < 1:
                    bbox_h = 1
                gt[idx, 2] = bbox_w / float(feature_stride)
                gt[idx, 3] = bbox_h / float(feature_stride)

            # 当前所有的gt和解码后的预测框之间计算iou, iou_gt_pred.shape=[nums_gt, nums_anchors]
            iou_gt_pred = bbox_ious(gt, cur_pred_boxes)
            # 和gt匹配大于0.6的位置掩码,这里注意, mask表示的是pre_bbox只要与所有的gt中,
            # 与某一个gt的iou>0.6,都设置为True
            ignore_thresh = self.train_cfg.get("neg_iou_thr", 0.6)
            # mask中有为True的地方表示: anchor与所有的gt中的iou的max值有大于ignore_thresh的情况.
            mask = (iou_gt_pred > ignore_thresh).sum(0) >= 1
            # 置为忽略样本.conf_neg_mask中1为负样本, 0为忽略样本.
            conf_neg_mask[b][mask.view_as(conf_neg_mask[b])] = 0

            # Find best anchor for each gt
            gt_wh = gt.clone()  # 每个gt,找出一个和他最匹配的anchor
            gt_wh[:, :2] = 0
            iou_gt_anchors = bbox_ious(gt_wh, anchors_wh)  # 不考虑gt位置，将gt和anchor的xy对齐计算
            _, best_anchors = iou_gt_anchors.max(1)  # 每一个gt匹配的最佳anchor,

            # Set masks and target values for each gt
            # time consuming
            for jdx, bbox in enumerate(gt_bboxs):  # 遍历每个gt,找到最匹配的anchor
                gi = min(nW - 1, max(0, int(gt[jdx, 0])))  # 找出该gt在第几个网格内部
                gj = min(nH - 1, max(0, int(gt[jdx, 1])))
                cur_n = best_anchors[jdx]  # 与第jdx个gt匹配的最佳的anchor下标
                if cur_n in anchors_mask:
                    best_n = anchors_mask.index(cur_n)  # 最好的anchor
                else:
                    continue
                # 对于yolov3, 如果在gt匹配到了最佳的anchor(且该anchor在分配的当前层内), 则才算一个正样本.
                all_pos_nums += 1
                # 得到匹配的label, 方便后面计算loss, 对于第b个样本, 在第(gj, gi)格子中,
                # 第b张中第gj*nW+gi位置的第best_n个anchor的coord_mask值(权重),
                # 正样本的权重大小为: (2 - gt_w * gt_h), 其中gt_w和gt_h为特征图尺度大小.
                coord_mask[b][best_n][0][gj * nW + gi] = 2 - bbox[2] * bbox[3] / (
                        nW * nH * feature_stride * feature_stride)  # 考虑小物体，加大权重
                # cls_mask[0][2][52]: 表示在第0号样本中, 在第52个格子中的第2个anchor在之后会计算分类损失权重
                cls_mask[b][best_n][gj * nW + gi] = 1
                # conf_pos_mask[0][2][52]: 表示在第0号样本中, 在第52个格子中的第2个anchor为正样本.
                conf_pos_mask[b][best_n][gj * nW + gi] = 1  # 有物体位置, 也即是该anchor为正样本.
                # conf_neg_mask[0][2][52]: 表示在第0号样本中, 在第52个格子中的第2个anchor设置为忽略
                conf_neg_mask[b][best_n][gj * nW + gi] = 0  # 对应在负样本mask中将位置设置为0.
                # tcoord[0][2][...][52]: 表示第0个样本中, 第52个格子中的第2个anchor的target坐标值
                tcoord[b][best_n][0][gj * nW + gi] = gt[jdx, 0] - gi  # 转换为相对于左上角偏移
                tcoord[b][best_n][1][gj * nW + gi] = gt[jdx, 1] - gj  # tcoord也表示target coord
                tcoord[b][best_n][2][gj * nW + gi] = math.log(gt[jdx, 2] / anchors[cur_n, 0])  # 转换为log
                tcoord[b][best_n][3][gj * nW + gi] = math.log(gt[jdx, 3] / anchors[cur_n, 1])
                # t_iou, 注意gt为特征图尺度.
                coord_iou[b][best_n][0][gj * nW + gi] = gt[jdx, 0]  # x
                coord_iou[b][best_n][1][gj * nW + gi] = gt[jdx, 1]  # y
                coord_iou[b][best_n][2][gj * nW + gi] = gt[jdx, 2]  # w
                coord_iou[b][best_n][3][gj * nW + gi] = gt[jdx, 3]  # h
                # tconf[0][2][52]表示第0个样本中, 第52个格子中的第2个anchor的IOU_k_truth目标值为iou(GT与预测框之间的iou)
                tconf[b][best_n][gj * nW + gi] = 1
                if nC > 1:
                    # tcls中第b个样本, 第(gj, gi)格子内的第best_n个anchor, 他的target_class_id为:int(bbox[4]) - 1
                    # one hot, 也即将bbox的类别填入到最佳anchor中
                    if self.one_hot_smoother != 0:  # label smooth
                        gt_label_one_hot = F.one_hot(gt_label[jdx], num_classes=self.num_classes).float()
                        gt_label_one_hot = gt_label_one_hot * (1 - self.one_hot_smoother
                                                               ) + self.one_hot_smoother / self.num_classes
                        tcls[b][best_n][gj * nW + gi] = gt_label_one_hot
                    else:
                        tcls[b][best_n][gj * nW + gi][int(gt_label[jdx])] = 1
                else:
                    tcls[b][best_n][gj * nW + gi] = 0

                # 这里我们除了考虑gt的最佳anchor, 我们也考虑gt所在cell内的其他当前层的anchor情况,
                # 这样其实能在一定程度上防止标签重写现象
                for kdx, anchor_mask in enumerate(anchors_mask):
                    if anchor_mask != cur_n:
                        iou = iou_gt_anchors[jdx][anchor_mask]
                        if iou >= self.train_cfg.get("pos_iou_thr", 0.2):
                            all_pos_nums += 1
                            coord_mask[b][kdx][0][gj * nW + gi] = 2 - bbox[2] * bbox[3] / (
                                    nW * nH * feature_stride * feature_stride)
                            cls_mask[b][kdx][gj * nW + gi] = 1
                            conf_pos_mask[b][kdx][gj * nW + gi] = 1
                            conf_neg_mask[b][kdx][gj * nW + gi] = 0

                            tcoord[b][kdx][0][gj * nW + gi] = gt[jdx, 0] - gi
                            tcoord[b][kdx][1][gj * nW + gi] = gt[jdx, 1] - gj
                            tcoord[b][kdx][2][gj * nW + gi] = math.log(gt[jdx, 2] / anchors[anchor_mask, 0])
                            tcoord[b][kdx][3][gj * nW + gi] = math.log(gt[jdx, 3] / anchors[anchor_mask, 1])

                            coord_iou[b][kdx][0][gj * nW + gi] = gt[jdx, 0]
                            coord_iou[b][kdx][1][gj * nW + gi] = gt[jdx, 1]
                            coord_iou[b][kdx][2][gj * nW + gi] = gt[jdx, 2]
                            coord_iou[b][kdx][3][gj * nW + gi] = gt[jdx, 3]

                            tconf[b][kdx][gj * nW + gi] = 1
                            if nC > 1:
                                if self.one_hot_smoother != 0:  # label smooth
                                    gt_label_one_hot = F.one_hot(gt_label[jdx], num_classes=self.num_classes).float()
                                    gt_label_one_hot = gt_label_one_hot * (1 - self.one_hot_smoother
                                                                           ) + self.one_hot_smoother / self.num_classes
                                    tcls[b][best_n][gj * nW + gi] = gt_label_one_hot
                                else:
                                    tcls[b][best_n][gj * nW + gi][int(gt_label[jdx])] = 1
                            else:
                                tcls[b][kdx][gj * nW + gi] = 0

        return [coord_mask, conf_pos_mask, conf_neg_mask,
                cls_mask, tcoord, tconf, tcls, coord_iou]

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

    def _get_bboxes_single(self, pred_maps_list, scale_factor,
                           cfg, rescale=False, with_nms=True):
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
        num_levels = len(pred_maps_list)
        featmap_sizes = [pred_maps_list[i].shape[-2:] for i in range(num_levels)]

        for idx, pred_map in enumerate(pred_maps_list):
            cur_anchors = self.anchors[self.anchors_mask[idx]]
            stride = self.strides[idx]
            map_h, map_w = featmap_sizes[idx]
            prediction = pred_map.view(len(cur_anchors), self.num_attrib, *featmap_sizes[idx])
            prediction = prediction.permute(0, 2, 3, 1).contiguous()
            # Get outputs
            x = torch.sigmoid(prediction[..., 0])  # Center x
            y = torch.sigmoid(prediction[..., 1])  # Center y
            w = prediction[..., 2]  # Width
            h = prediction[..., 3]  # Height
            pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
            # TODO: consider sigmoid or softmax of mode for cls as config
            pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

            # generate offset for each grid
            FloatTensor = torch.cuda.FloatTensor if prediction.is_cuda else torch.FloatTensor
            grid_x = torch.arange(map_w).repeat(map_h, 1).view([1, map_h, map_w]).type(FloatTensor)
            grid_y = torch.arange(map_h).unsqueeze(0).t().repeat(1, map_w).view([1, map_h, map_w]).type(FloatTensor)
            scaled_anchors = FloatTensor([(a_w / stride, a_h / stride) for a_w, a_h in cur_anchors])
            anchor_w = scaled_anchors[:, 0:1].view((len(cur_anchors), 1, 1))
            anchor_h = scaled_anchors[:, 1:2].view((len(cur_anchors), 1, 1))

            # decode to orig size
            pred_boxes = FloatTensor(prediction[..., :4].shape)
            pred_boxes[..., 0] = (x.data + grid_x) * stride  # x
            pred_boxes[..., 1] = (y.data + grid_y) * stride  # y
            pred_boxes[..., 2] = torch.exp(w.data) * anchor_w * stride  # w
            pred_boxes[..., 3] = torch.exp(h.data) * anchor_h * stride  # h

            # reshape
            pred_boxes = xywh2xyxy(pred_boxes.view(-1, 4))
            conf_pred = pred_conf.view(-1)
            cls_pred = pred_cls.view(-1, self.num_classes)  # Cls pred one-hot.

            # Filtering out all predictions with conf < conf_thr
            conf_thr = cfg.get('conf_thr', -1)
            conf_inds = conf_pred.ge(conf_thr).nonzero().flatten()
            bbox_pred = pred_boxes[conf_inds, :]
            cls_pred = cls_pred[conf_inds, :]
            conf_pred = conf_pred[conf_inds]

            # Get top-k prediction
            nms_pre = cfg.get('nms_pre', -1)
            if 0 < nms_pre < conf_pred.size(0) and (
                    not torch.onnx.is_in_onnx_export()):
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

    def aug_test(self, feats, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            feats (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains features for all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[ndarray]: bbox results of each class
        """
        return self.aug_test_bboxes(feats, img_metas, rescale=rescale)
