from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import get_dist_info
from detectron2.layers import ShapeSpec
from detectron2.structures import Boxes

from openselfsup.models.utils.d2_fpn import build_resnet_fpn_backbone
from openselfsup.models.necks import _init_weights

from openselfsup.utils import print_log
from openselfsup.models.registry import MODELS
from openselfsup.datasets.pipelines.kornia_transform import AugmentationPipeline

from openselfsup.models.utils.pooler import ROIPooler

import torch.cuda

INF = 1e12

class ContrastiveHead(nn.Module):
    """Head for contrastive learning.

    Args:
        temperature (float): The temperature hyper-parameter that
            controls the concentration level of the distribution.
            Default: 0.1.
    """

    def __init__(self, temperature=0.2):
        super(ContrastiveHead, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.temperature = temperature

    def forward(self, pos, neg):
        """Forward head.

        Args:
            pos (Tensor): Nx1 positive similarity.
            neg (Tensor): Nxk negative similarity.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        N = pos.size(0)
        logits = torch.cat((pos, neg), dim=1)
        logits /= self.temperature
        labels = torch.zeros((N, ), dtype=torch.long).cuda()
        loss = self.criterion(logits, labels)
        return loss

class LinearNormNeck(nn.Module):
    def __init__(
            self,
            in_channels,
            hid_channels,
            out_channels,
            num_mlp,
            with_avg_pool=False,
            norm='',
            last_norm=True,
            dropout_p=0.1
    ):
        super(LinearNormNeck, self).__init__()
        assert norm == 'SyncBN' or norm == '' or norm == 'BN'
        self.with_avg_pool = with_avg_pool
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        mlps = OrderedDict()
        cur_channels = in_channels
        for i in range(num_mlp):
            mlps['linear_%d'%(i+1)] = nn.Sequential(nn.Linear(cur_channels, hid_channels, bias=norm==''), nn.Dropout(p=dropout_p))
            if norm == 'SyncBN':
                mlps['norm_%d'%(i+1)] = nn.SyncBatchNorm(num_features=hid_channels)
            elif norm == 'BN':
                mlps['norm_%d' % (i + 1)] = nn.BatchNorm1d(num_features=hid_channels)
            mlps['relu_%d'%(i+1)] = nn.ReLU(inplace=True)
            cur_channels = hid_channels
        mlps['last_linear'] = nn.Linear(cur_channels, out_channels, bias=not last_norm)
        if last_norm:
            if norm == 'SyncBN':
                mlps['last_norm'] = nn.SyncBatchNorm(num_features=out_channels, affine=False)
            elif norm == 'BN':
                mlps['last_norm'] = nn.BatchNorm1d(num_features=out_channels, affine=False)
        self.mlp = nn.Sequential(mlps)

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)

    def forward(self, x):
        if self.with_avg_pool:
            x = self.avgpool(x)
        return self.mlp(x.view(x.size(0), -1))

class MultiPooler(nn.Module):
    def __init__(self, output_sizes, pooler_scales, sampling_ratio, assign_sizes, with_avg_pool, flatten):
        super(MultiPooler, self).__init__()
        assert isinstance(output_sizes, tuple)
        assert len(output_sizes) == len(assign_sizes)

        if len(output_sizes) == 1:
            self.roi_poolers = ROIPooler(
                output_size=output_sizes[0],
                scales=pooler_scales,
                sampling_ratio=sampling_ratio,
                pooler_type='ROIAlignV2',
            )
            self.assign_sizes = assign_sizes[0]
            self.single = True
        else:
            self.roi_poolers = nn.ModuleList()
            for i, size in enumerate(output_sizes):
                self.roi_poolers.append(
                    ROIPooler(
                        output_size=size,
                        scales=pooler_scales,
                        sampling_ratio=sampling_ratio,
                        pooler_type='ROIAlignV2',
                    )
                )
            self.assign_sizes = assign_sizes
            self.single = False
        self.with_avg_pool = with_avg_pool
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = flatten

    def forward(self, feats, boxes):
        n = sum([len(x) for x in boxes])
        if self.single:
            roi_feats = self.roi_poolers(feats, boxes, soco_assign_sizes=self.assign_sizes)
            if self.with_avg_pool:
                roi_feats = self.avgpool(roi_feats).view(n, -1)
            else:
                if self.flatten:
                    roi_feats = roi_feats.view(n, -1)
        else:
            roi_feats = [pooler(feats, boxes, soco_assign_sizes=sizes)
                         for pooler, sizes in zip(self.roi_poolers, self.assign_sizes)]
            if self.with_avg_pool:
                roi_feats = [self.avgpool(x) for x in roi_feats]
                roi_feats = torch.cat(roi_feats, dim=1).view(n, -1)
            else:
                if self.flatten:
                    roi_feats = torch.cat([x.flatten(1, 3) for x in roi_feats], dim=1)
                else:
                    roi_feats = torch.cat(roi_feats, dim=1)
        return roi_feats

    def get_output_channels(self, in_channels):
        if self.with_avg_pool:
            if self.single:
                return in_channels
            else:
                return in_channels * len(self.roi_poolers)
        else:
            if self.single:
                if self.flatten:
                    return in_channels * self.roi_poolers.output_size[0] ** 2
                else:
                    return in_channels
            else:
                if self.flatten:
                    return in_channels * sum(map(lambda x: x.output_size[0]**2, self.roi_poolers))
                else:
                    return in_channels * len(self.roi_poolers)

def get_iou_mat(boxes):
    with torch.no_grad():
        if type(boxes) is Boxes:
            boxes_ = boxes.tensor
        else:
            boxes_ = boxes.clone()

        n_boxes = boxes_.size(0)

        boxes_1 = boxes_[:, None, :].repeat(1, n_boxes, 1)  # [n_boxes, n_jitter, 4]
        boxes_2 = boxes_1.permute(1, 0, 2) # [n_boxes, n_jitter, 4]

        xmin1 = boxes_1[:, :, 0]
        ymin1 = boxes_1[:, :, 1]
        xmax1 = boxes_1[:, :, 2]
        ymax1 = boxes_1[:, :, 3]

        xmin2 = boxes_2[:, :, 0]
        ymin2 = boxes_2[:, :, 1]
        xmax2 = boxes_2[:, :, 2]
        ymax2 = boxes_2[:, :, 3]

        xmin_int = torch.max(xmin1, xmin2)
        ymin_int = torch.max(ymin1, ymin2)
        xmax_int = torch.min(xmax1, xmax2)
        ymax_int = torch.min(ymax1, ymax2)

        w_int = (xmax_int - xmin_int).clamp(min=0.)
        h_int = (ymax_int - ymin_int).clamp(min=0.)

        area_int = w_int * h_int
        area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
        area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

        iou = area_int / (area1 + area2 - area_int)
        return iou

def intra_loss(sim_by_img, valid_by_img, margin):
    triplet_by_img = [F.relu(x - x.diag()[None, :] + margin).view(-1) for x in sim_by_img]
    triplet_all = torch.cat(triplet_by_img)
    valids = torch.cat([x.view(-1) for x in valid_by_img])

    triplet_loss = triplet_all[valids].mean()
    return triplet_loss

def aug_boxes(img_shape, boxes, n_jitter, xy_mag, wh_mag, iou_range, append_ori):
    with torch.no_grad():
        x1 = boxes.tensor[:, 0]
        y1 = boxes.tensor[:, 1]
        x2 = boxes.tensor[:, 2]
        y2 = boxes.tensor[:, 3]

        x = (x1 + x2) * 0.5
        y = (y1 + y2) * 0.5
        w = x2 - x1
        h = y2 - y1

        t_xy = (torch.rand(size=(len(boxes), n_jitter, 2), device=x1.device) - 0.5) * xy_mag  # [N, n_jitter,2]
        t_wh = (torch.rand(size=(len(boxes), n_jitter, 2), device=x1.device) - 0.5) * wh_mag  # [N, n_jitter,2]

        x_jitter = w[:, None] * t_xy[:, :, 0] + x[:, None]
        y_jitter = h[:, None] * t_xy[:, :, 1] + y[:, None]
        w_jitter = torch.exp(t_wh[:, :, 0]) * w[:, None]
        h_jitter = torch.exp(t_wh[:, :, 1]) * h[:, None]

        x1_jitter = x_jitter - 0.5 * w_jitter
        y1_jitter = y_jitter - 0.5 * h_jitter
        x2_jitter = x_jitter + 0.5 * w_jitter
        y2_jitter = y_jitter + 0.5 * h_jitter

        x1_jitter = x1_jitter.clamp(min=0, max=img_shape[1])
        y1_jitter = y1_jitter.clamp(min=0, max=img_shape[0])
        x2_jitter = x2_jitter.clamp(min=0, max=img_shape[1])
        y2_jitter = y2_jitter.clamp(min=0, max=img_shape[0])

        jit_boxes = torch.stack((x1_jitter, y1_jitter, x2_jitter, y2_jitter), dim=2)  # [N, n_jitter, 4]
        if append_ori:
            jit_boxes = torch.cat((jit_boxes, boxes.tensor.view(-1, 1, 4)), dim=1)  # [N, n_jitter+1, 4]
        box_jitter_iou_mat = get_jitter_iou_mat(boxes, jit_boxes)  # [N, n_jitter+1]
        invalid = (box_jitter_iou_mat < iou_range[0])
        return Boxes(jit_boxes.view(-1, 4)), invalid

def get_jitter_iou_mat(boxes, jitter):
    with torch.no_grad():
        if type(boxes) is Boxes:
            boxes = boxes.tensor
        if type(jitter) is Boxes:
            jitter = jitter.tensor

        n_jitter = jitter.size(1)

        boxes_ = boxes[:, None, :].repeat(1, n_jitter, 1)  # [N, n_jitter, 4]

        xmin1 = boxes_[:, :, 0]
        ymin1 = boxes_[:, :, 1]
        xmax1 = boxes_[:, :, 2]
        ymax1 = boxes_[:, :, 3]

        xmin2 = jitter[:, :, 0]
        ymin2 = jitter[:, :, 1]
        xmax2 = jitter[:, :, 2]
        ymax2 = jitter[:, :, 3]

        xmin_int = torch.max(xmin1, xmin2)
        ymin_int = torch.max(ymin1, ymin2)
        xmax_int = torch.min(xmax1, xmax2)
        ymax_int = torch.min(ymax1, ymax2)

        w_int = (xmax_int - xmin_int).clamp(min=0.)
        h_int = (ymax_int - ymin_int).clamp(min=0.)

        area_int = w_int * h_int
        area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
        area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

        iou = area_int / (area1 + area2 - area_int)  # [N, n_jitter]
        return iou

@MODELS.register_module
class CCOP(nn.Module):
    def __init__(self,
                 backbone_cfg,
                 pretrained,
                 img_neck_params,
                 roi_neck_params,
                 img_queue_len=65536,
                 roi_queue_len=65536,
                 momentum=0.999,
                 img_temperature=0.2,
                 roi_temperature=0.2,
                 pooling_sizes=(7,),
                 pooling_assign_sizes=((224, 192, 96, 48),),
                 training=True,
                 aug_dicts=None,
                 curr_params=None,
                 total_step=1.,
                 **kwargs):

        super(CCOP, self).__init__()

        # Network q and RoI Pooler
        self.backbone_q = build_resnet_fpn_backbone(backbone_cfg, ShapeSpec(channels=len(backbone_cfg.MODEL.PIXEL_MEAN)))
        self.img_neck_q = LinearNormNeck(
            img_neck_params['in_channels'],
            img_neck_params['hid_channels'],
            img_neck_params['embed_channels'],
            img_neck_params['num_mlp'],
            img_neck_params['with_avg_pool'],
            img_neck_params['norm'],
            img_neck_params['last_bn']
        )

        input_shape = self.backbone_q.output_shape()
        in_features = backbone_cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio = backbone_cfg.MODEL.ROI_HEADS.POOLER_SAMPLING_RATIO
        self.roi_pooler = MultiPooler(
            output_sizes=pooling_sizes,
            pooler_scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            assign_sizes=pooling_assign_sizes,
            with_avg_pool=roi_neck_params['with_avg_pool'],
            flatten=True
        )

        roi_neck_in_channels = self.roi_pooler.get_output_channels(roi_neck_params['in_channels'])
        self.roi_neck_q = LinearNormNeck(
        roi_neck_in_channels,
        roi_neck_params['hid_channels'],
        roi_neck_params['embed_channels'],
        roi_neck_params['num_mlp'],
        False,  # avgpool already applied in Pooler
        roi_neck_params['norm'],
        roi_neck_params['last_bn']
        )

        # Network k
        self.backbone_k = build_resnet_fpn_backbone(backbone_cfg, ShapeSpec(channels=len(backbone_cfg.MODEL.PIXEL_MEAN)))
        self.img_neck_k = LinearNormNeck(
            img_neck_params['in_channels'],
            img_neck_params['hid_channels'],
            img_neck_params['embed_channels'],
            img_neck_params['num_mlp'],
            img_neck_params['with_avg_pool'],
            img_neck_params['norm'],
            img_neck_params['last_bn']
        )
        self.roi_neck_k = LinearNormNeck(
            roi_neck_in_channels,
            roi_neck_params['hid_channels'],
            roi_neck_params['embed_channels'],
            roi_neck_params['num_mlp'],
            False,  # avgpool already applied in Pooler
            roi_neck_params['norm'],
            roi_neck_params['last_bn']
        )

        # For loss computing
        self.img_contrastive_head = ContrastiveHead(temperature=img_temperature)
        self.roi_contrastive_head = ContrastiveHead(temperature=roi_temperature)

        # Parameter init
        for param in self.backbone_k.parameters():
            param.requires_grad = False
        for param in self.img_neck_k.parameters():
            param.requires_grad = False
        for param in self.roi_neck_k.parameters():
            param.requires_grad = False
        self.init_weights(pretrained)

        # Queue init
        self.img_queue_len = img_queue_len
        self.roi_queue_len = roi_queue_len

        self.register_buffer("img_queue", torch.randn(img_neck_params['embed_channels'], img_queue_len))
        self.register_buffer("roi_queue", torch.randn(roi_neck_params['embed_channels'], roi_queue_len))

        self.img_queue = nn.functional.normalize(self.img_queue, dim=0)
        self.roi_queue = nn.functional.normalize(self.roi_queue, dim=0)

        self.register_buffer("img_queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("roi_queue_ptr", torch.zeros(1, dtype=torch.long))

        # others
        self.roi_embed_dim = roi_neck_params['embed_channels']
        self.momentum = momentum
        self.rcnn_in_feature = backbone_cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.register_buffer("pixel_mean", torch.Tensor(backbone_cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(backbone_cfg.MODEL.PIXEL_STD).view(-1, 1, 1))

        # data augmentation
        if aug_dicts is not None:
            self.pipeline_v1 = AugmentationPipeline(aug_dicts[0])
            self.pipeline_v2 = AugmentationPipeline(aug_dicts[1])
            self.in_model_aug = True
        else:
            self.in_model_aug = False


        self.curr_params = curr_params
        self.register_buffer("iter", torch.zeros(1, dtype=torch.long))
        self.total_step = total_step   

    def preprocess_image(self, img):
        if len(img.shape) == 5:
            im_q = img[:, 0, ...].contiguous()
            im_k = img[:, 1, ...].contiguous()
            if self.in_model_aug:
                im_q = self.pipeline_v1(im_q)
                im_k = self.pipeline_v2(im_k)
            im_q *= 255.
            im_k *= 255.
            im_q = (im_q - self.pixel_mean) / self.pixel_std
            im_k = (im_k - self.pixel_mean) / self.pixel_std
            return im_q, im_k
        else:
            img_input = img.contiguous() * 255.
            return (img_input - self.pixel_mean) / self.pixel_std

    def init_weights(self, pretrained=None):
        assert pretrained is None
        if pretrained is not None:
            print_log('load model from: {}'.format(pretrained), logger='root')

        self.img_neck_q.init_weights()
        self.roi_neck_q.init_weights()

        for param_q, param_k in zip(self.backbone_q.parameters(),
                                    self.backbone_k.parameters()):
            param_k.data.copy_(param_q.data)

        for param_q, param_k in zip(self.img_neck_q.parameters(),
                                    self.img_neck_k.parameters()):
            param_k.data.copy_(param_q.data)

        for param_q, param_k in zip(self.roi_neck_q.parameters(),
                                    self.roi_neck_k.parameters()):
            param_k.data.copy_(param_q.data)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder."""
        for param_q, param_k in zip(self.backbone_q.parameters(),
                                    self.backbone_k.parameters()):
            param_k.data = param_k.data * self.momentum + \
                           param_q.data * (1. - self.momentum)

        for param_q, param_k in zip(self.img_neck_q.parameters(),
                                    self.img_neck_k.parameters()):
            param_k.data = param_k.data * self.momentum + \
                           param_q.data * (1. - self.momentum)

        for param_q, param_k in zip(self.roi_neck_q.parameters(),
                                    self.roi_neck_k.parameters()):
            param_k.data = param_k.data * self.momentum + \
                           param_q.data * (1. - self.momentum)

    @torch.no_grad()
    def _roi_dequeue_and_enqueue(self, keys):
        """Update queue."""
        # gather keys before updating queue

        numbers = keys.new_full((1,), 0, dtype=torch.int64) + len(keys)
        numbers_all = concat_all_gather(numbers)
        max_number = numbers_all.max().item()

        keys = concat_all_gather_mutable_length(keys, numbers, max_number)

        batch_size = keys.shape[0]

        ptr = int(self.roi_queue_ptr)

        valid_batch_size = min(batch_size, self.roi_queue_len - ptr)
        in_keys = keys[:valid_batch_size]

        self.roi_queue[:, ptr:ptr + valid_batch_size] = in_keys.transpose(0, 1)
        ptr = (ptr + valid_batch_size) % self.roi_queue_len  # move pointer

        self.roi_queue_ptr[0] = ptr

    @torch.no_grad()
    def _img_dequeue_and_enqueue(self, keys):
        """Update queue."""
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.img_queue_ptr)
        assert self.img_queue_len % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.img_queue[:, ptr:ptr + batch_size] = keys.transpose(0, 1)
        ptr = (ptr + batch_size) % self.img_queue_len  # move pointer

        self.img_queue_ptr[0] = ptr

    @torch.no_grad()
    def _img_batch_shuffle_ddp(self, x):
        """Batch shuffle, for making use of BatchNorm.

        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """Undo batch shuffle.

        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward_train(self, img, boxes, boxes_num, **kwargs):
        assert img.dim() == 5, "Input must have 5 dims, got: {}".format(img.dim())
        self.iter += 1

        im_q, im_k = self.preprocess_image(img)

        # no box jit here
        boxes_num = boxes_num.reshape(-1)
        boxes_q = []
        boxes_k = []
        for i in range(len(boxes)):
            boxes_q.append(Boxes(boxes[i][0][:boxes_num[i]]))
            boxes_k.append(Boxes(boxes[i][1][:boxes_num[i]]))

        img_feat_q, feats_q = self.backbone_q(im_q, ret_img_feature=True)
        img_embed_q = self.img_neck_q(img_feat_q)
        img_embed_q = F.normalize(img_embed_q, dim=1)

        feats_q = [feats_q[f] for f in self.rcnn_in_feature]
        roi_feats_q = self.roi_pooler(feats_q, boxes_q)
        roi_embed_q = self.roi_neck_q(roi_feats_q)
        roi_embed_q = F.normalize(roi_embed_q, dim=1)

        roi_embed_q_byImg = [roi_embed_q[boxes_num[:i].sum():boxes_num[:(i + 1)].sum()] for i in range(len(im_q))]

        # compute key features
        with torch.no_grad():
            self._momentum_update_key_encoder()

            im_k, idx_unshuffle = self._img_batch_shuffle_ddp(im_k)

            img_feat_k, feats_k = self.backbone_k(im_k, ret_img_feature=True)
            img_embed_k = self.img_neck_k(img_feat_k)
            img_embed_k = F.normalize(img_embed_k, dim=1)
            img_embed_k = self._batch_unshuffle_ddp(img_embed_k, idx_unshuffle)

            feats_k = [feats_k[f] for f in self.rcnn_in_feature]
            feats_k = [self._batch_unshuffle_ddp(x, idx_unshuffle) for x in feats_k]

            n_jitter = self.curr_params['n_jitter']
            iou_thr = self.curr_params['iou_thr'][0] + (self.iter / self.total_step) * (self.curr_params['iou_thr'][1] - self.curr_params['iou_thr'][0])
            mag_xy = self.curr_params['jit_mag'][0] + (self.iter / self.total_step) * (self.curr_params['jit_mag'][1] - self.curr_params['jit_mag'][0])
            mag_wh = self.curr_params['jit_mag'][2] + (self.iter / self.total_step) * (self.curr_params['jit_mag'][3] - self.curr_params['jit_mag'][2])

            aug_boxes_k = []
            invalid_mats_k = []
            for box_per_img in boxes_k:
                box_jit_per_img, invalid_per_img = aug_boxes(im_q.shape[-2:], box_per_img, n_jitter, mag_xy, mag_wh, (iou_thr, INF), append_ori=True)
                aug_boxes_k.append(box_jit_per_img)  # [n_box * (n_jitter+1), 4]
                invalid_mats_k.append(invalid_per_img)  # [n_box, n_jitter+1]
            invalid_mats_k = torch.cat(invalid_mats_k, dim=0)

            roi_feats_k = self.roi_pooler(feats_k, aug_boxes_k)
            roi_embed_k = self.roi_neck_k(roi_feats_k)  # [n_box * (n_jitter+1), n_dim]
            roi_embed_k = F.normalize(roi_embed_k, dim=1)

            roi_embed_k = roi_embed_k.reshape(-1, n_jitter + 1, self.roi_embed_dim)
            q_k_sim = (roi_embed_q.view(-1, 1, self.roi_embed_dim) * roi_embed_k).sum(dim=-1)  # [n_img * n_box, n_jitter+1]
            q_k_sim = q_k_sim + (invalid_mats_k * 5.)  # make sure invalid similarity is larger than 1 so it will never be selected
            min_inds = torch.argmin(q_k_sim, dim=1)

            roi_pos_embed = roi_embed_k[torch.arange(roi_embed_k.size(0)), min_inds]

            roi_feats_k_ori = self.roi_pooler(feats_k, boxes_k)
            roi_feats_k_ori = self.roi_neck_k(roi_feats_k_ori)  # [n_box * (n_jitter+1), n_dim]
            roi_feats_k_ori = F.normalize(roi_feats_k_ori, dim=1)

            inter_valid = [(get_iou_mat(x) < 0.05).detach_() for x in boxes_k]
            roi_embed_k_byImg = [roi_feats_k_ori[boxes_num[:i].sum():boxes_num[:(i + 1)].sum()] for i in range(len(im_k))]

        img_pos = (img_embed_q * img_embed_k).sum(dim=1).reshape(-1, 1)
        roi_pos = (roi_embed_q * roi_pos_embed).sum(dim=1).reshape(-1, 1)

        img_neg = torch.matmul(img_embed_q, self.img_queue.clone().detach())
        roi_neg = torch.matmul(roi_embed_q, self.roi_queue.clone().detach())

        sim_inImg = [torch.matmul(x, y.transpose(0, 1)) for (x, y) in zip(roi_embed_q_byImg, roi_embed_k_byImg)]
        loss_inImg = intra_loss(sim_inImg, inter_valid, margin=0.4)

        img_loss = self.img_contrastive_head(img_pos, img_neg)
        roi_loss = self.roi_contrastive_head(roi_pos, roi_neg)

        self._img_dequeue_and_enqueue(img_embed_k)
        self._roi_dequeue_and_enqueue(roi_pos_embed)
        return {'img_loss': img_loss, 'roi_loss': roi_loss, 'inImg_loss': loss_inImg}

    def forward_test(self, img, **kwargs):
        return None

    def forward(self, img, mode='train', **kwargs):
        if mode == 'train':
          return self.forward_train(img, **kwargs)
        elif mode == 'test':
            return self.forward_test(img, **kwargs)
        elif mode == 'extract':
            return self.backbone_q(img)
        else:
            raise Exception("No such mode: {}".format(mode))


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """Performs all_gather operation on the provided tensors.

    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

def concat_all_gather_mutable_length(tensor, number, max_number):
    tensor_shape = tensor.shape
    tensors_gather = [
        torch.ones((max_number, *tensor_shape[1:]), dtype=tensor.dtype, device=tensor.device)
        for _ in range(torch.distributed.get_world_size())
    ]
    numbers_gather = [
        torch.ones_like(number) for _ in range(torch.distributed.get_world_size())
    ]

    pad = [0 for _ in range(len(tensor_shape) * 2)]
    pad[-1] = max_number - tensor.size(0)
    tensor_pad = F.pad(tensor, pad)

    torch.distributed.all_gather(tensors_gather, tensor_pad, async_op=False)
    torch.distributed.all_gather(numbers_gather, number, async_op=False)

    all_numbers = torch.cat(numbers_gather, dim=0)
    all_tensors = torch.cat([tensors_gather[i][:all_numbers[i]] for i in range(torch.distributed.get_world_size())], dim=0)
    return all_tensors
