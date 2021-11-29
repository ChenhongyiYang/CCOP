import torch
from PIL import Image

from openselfsup.datasets.registry import DATASETS
from openselfsup.datasets.base import BaseDataset

import math
import copy
import random
import logging
import numpy as np
from typing import List, Optional, Union
import torch
import torchvision.transforms.transforms
from torchvision.transforms import RandomResizedCrop
import torchvision.transforms.functional as TVF
from PIL import Image

from detectron2.structures import Boxes

def RandomResizedCrop_get_params(width, height, scale: List[float], ratio: List[float]):
    """Get parameters for ``crop`` for a random sized crop.

    Args:
        img (PIL Image or Tensor): Input image.
        scale (list): range of scale of the origin size cropped
        ratio (list): range of aspect ratio of the origin aspect ratio cropped

    Returns:
        tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop.
    """
    area = height * width
    for _ in range(10):
        target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
        log_ratio = torch.log(torch.tensor(ratio))
        aspect_ratio = torch.exp(
            torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
        ).item()

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        if 0 < w <= width and 0 < h <= height:
            i = torch.randint(0, height - h + 1, size=(1,)).item()
            j = torch.randint(0, width - w + 1, size=(1,)).item()
            return i, j, h, w

    # Fallback to central crop
    in_ratio = float(width) / float(height)
    if in_ratio < min(ratio):
        w = width
        h = int(round(w / min(ratio)))
    elif in_ratio > max(ratio):
        h = height
        w = int(round(h * max(ratio)))
    else:  # whole image
        w = width
        h = height
    i = (height - h) // 2
    j = (width - w) // 2
    return i, j, h, w

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

class AdvancedInstanceContrastAug(object):
    def __init__(self, size, scale, ratio, area_thr, part_thr, min_box_num, max_try):
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.small_thr = area_thr
        self.part_thr = part_thr
        self.min_box_num = min_box_num
        self.max_try = max_try

    def _filter_out_boxes(self, boxes, ori_areas):
        # filter out small (zero) boxes
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        # filter out partial boxes
        part_keep = (areas / ori_areas) > self.part_thr
        return part_keep

    def _one_try(self, img, boxes, scale):
        if scale is None:
            top, left, h, w = RandomResizedCrop.get_params(img, self.scale, self.ratio)
        else:
            top, left, h, w = RandomResizedCrop.get_params(img, scale, self.ratio)

        new_boxes = boxes.clone()
        new_boxes[:, 0] = (new_boxes[:, 0] - left).clamp(min=0, max=w)
        new_boxes[:, 1] = (new_boxes[:, 1] - top).clamp(min=0, max=h)
        new_boxes[:, 2] = (new_boxes[:, 2] - left).clamp(min=0, max=w)
        new_boxes[:, 3] = (new_boxes[:, 3] - top).clamp(min=0, max=h)

        return (top, left, h, w), new_boxes

    def _resize_box(self, boxes, h, w):
        h = float(h)
        w = float(w)

        boxes[:, 0] *= self.size / w
        boxes[:, 1] *= self.size / h
        boxes[:, 2] *= self.size / w
        boxes[:, 3] *= self.size / h

        return boxes

    def apply_random_hflip(self, img, boxes):
        _img = img.copy()
        _boxes = boxes.clone()
        width, height = _img.size
        r = random.random()
        if r > 0.5:
            _img = _img.transpose(Image.FLIP_LEFT_RIGHT)
            _boxes[:, 0] = width - boxes[:, 2]
            _boxes[:, 2] = width - boxes[:, 0]

        return _img, _boxes

    def apply_box_jitter(self, boxes, xy_mag, wh_mag, n_jitter, iou_range):
        assert len(iou_range) == 2
        assert iou_range[0] < iou_range[1]

        if len(boxes) == 0:
            return boxes

        with torch.no_grad():
            x1 = boxes[:, 0]
            y1 = boxes[:, 1]
            x2 = boxes[:, 2]
            y2 = boxes[:, 3]

            x = (x1 + x2) * 0.5
            y = (y1 + y2) * 0.5
            w = x2 - x1
            h = y2 - y1

            t_xy = (torch.rand(size=(len(boxes), n_jitter, 2)) - 0.5) * xy_mag  # [N, n_jitter,2]
            t_wh = (torch.rand(size=(len(boxes), n_jitter, 2)) - 0.5) * wh_mag  # [N, n_jitter,2]

            x_jitter = w[:, None] * t_xy[:, :, 0] + x[:, None]
            y_jitter = h[:, None] * t_xy[:, :, 1] + y[:, None]
            w_jitter = torch.exp(t_wh[:, :, 0]) * w[:, None]
            h_jitter = torch.exp(t_wh[:, :, 1]) * h[:, None]

            x1_jitter = x_jitter - 0.5 * w_jitter
            y1_jitter = y_jitter - 0.5 * h_jitter
            x2_jitter = x_jitter + 0.5 * w_jitter
            y2_jitter = y_jitter + 0.5 * h_jitter

            jit_boxes = torch.stack((x1_jitter, y1_jitter, x2_jitter, y2_jitter), dim=2)  # [N, n_jitter, 4]
            box_jitter_iou_mat = get_jitter_iou_mat(boxes, jit_boxes)  # [N, n_jitter]

            final_boxes = []
            for i in range(len(jit_boxes)):
                iou_vec = box_jitter_iou_mat[i]
                valid_inds = torch.arange(n_jitter, dtype=torch.long)[(iou_vec >= iou_range[0]) & (iou_vec <= iou_range[1])]
                if len(valid_inds) == 0:
                    final_boxes.append(jit_boxes[i][0])
                else:
                    selected_ind = valid_inds[torch.randperm(len(valid_inds))[0]]
                    final_boxes.append(jit_boxes[i][selected_ind])
            return torch.stack(final_boxes, dim=0)

    def apply_aug(self, img, boxes, scale=None, resize=True, ret_keep=False):
        num_boxes = len(boxes)
        ori_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        min_box_num = min(num_boxes, self.min_box_num)
        current_max = 0
        cur_crop_params = None
        cur_keep = None
        cur_boxes = None
        for i in range(self.max_try):
            crop_params, new_boxes = self._one_try(img, boxes, scale)
            keep = self._filter_out_boxes(new_boxes, ori_areas)
            n = keep.int().sum()
            if n >= min_box_num:
                y, x, h, w = crop_params
                if not ret_keep:
                    new_boxes = new_boxes[keep]
                if resize:
                    new_img = TVF.resized_crop(img, y, x, h, w, [self.size, self.size])
                    new_boxes = self._resize_box(new_boxes, h, w)
                else:
                    new_img = TVF.crop(img, y, x, h, w)
                im_w, im_h = img.size
                scale = np.sqrt(w * h) / np.sqrt(im_w * im_h)
                if not ret_keep:
                    return new_img, new_boxes, scale
                else:
                    return new_img, new_boxes, keep, scale

            if n >= current_max:
                cur_crop_params = crop_params
                cur_keep = keep
                cur_boxes = new_boxes
                current_max = n

        if not ret_keep:
            new_boxes = cur_boxes[cur_keep]
        else:
            new_boxes = cur_boxes

        y, x, h, w = cur_crop_params
        if resize:
            new_img = TVF.resized_crop(img, y, x, h, w, [self.size, self.size])
            new_boxes = self._resize_box(new_boxes, h, w)
        else:
            new_img = TVF.crop(img, y, x, h, w)
        im_w, im_h = img.size
        scale = np.sqrt(w * h) / np.sqrt(im_w * im_h)

        if not ret_keep:
            return new_img, new_boxes, scale
        else:
            return new_img, new_boxes, cur_keep, scale

    def __call__(self, img, instances):
        cropped_img, new_boxes, scale = self.apply_aug(img, instances.gt_boxes.tensor)

        img1, boxes1 = self.apply_random_hflip(cropped_img, new_boxes)
        img2, boxes2 = self.apply_random_hflip(cropped_img, new_boxes)

        boxes1 = self.apply_box_jitter(boxes1, 1., 1., n_jitter=5, iou_range=(0.3, 0.6))
        boxes2 = self.apply_box_jitter(boxes2, 1., 1., n_jitter=5, iou_range=(0.3, 0.6))

        return img1, img2, boxes1, boxes2

class ScaleInstanceContrastAug(object):
    def __init__(self, size, scales, ratio, area_thr, part_thr, min_box_num, max_try):
        self.aug = AdvancedInstanceContrastAug(size, (0.2, 1.), ratio, area_thr, part_thr, min_box_num, max_try)
        self.aug = AdvancedInstanceContrastAug(size, (0.2, 1.), ratio, area_thr, part_thr, min_box_num, max_try)
        self.small_thr = area_thr
        self.scales = scales

    def __call__(self, img, instances, ret_class=False):

        contrast_boxes = instances.gt_boxes.tensor


        img1, boxes1, keep1, _ = self.aug.apply_aug(img, contrast_boxes, scale=self.scales[0], resize=True, ret_keep=True)
        img2, boxes2, keep2, _ = self.aug.apply_aug(img, contrast_boxes, scale=self.scales[1], resize=True, ret_keep=True)
        classes = instances.gt_classes

        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

        area_keep = (area1 > self.small_thr) & (area2 > self.small_thr)

        boxes1 = boxes1[keep1 & keep2 & area_keep]
        boxes2 = boxes2[keep1 & keep2 & area_keep]
        classes = classes[keep1 & keep2 & area_keep]

        img1, boxes1 = self.aug.apply_random_hflip(img1, boxes1)
        img2, boxes2 = self.aug.apply_random_hflip(img2, boxes2)

        r = random.random()
        if r > 0.5:
            boxes1 = boxes1.clamp(min=0, max=self.aug.size - 1)
            boxes2 = boxes2.clamp(min=0, max=self.aug.size - 1)
            if ret_class:
                return img1, img2, boxes1, boxes2, classes
            else:
                return img1, img2, boxes1, boxes2
        else:
            boxes1 = boxes1.clamp(min=0, max=self.aug.size - 1)
            boxes2 = boxes2.clamp(min=0, max=self.aug.size - 1)
            if ret_class:
                return img2, img1, boxes2, boxes1, classes
            else:
                return img2, img1, boxes2, boxes1



@DATASETS.register_module
class ContrastiveBox(BaseDataset):
    """Dataset for contrastive learning methods that forward
        two views of the image at a time (MoCo, SimCLR).
    """

    def __init__(self, data_source, pipeline, aug_dict, prefetch = False):
        super(ContrastiveBox, self).__init__(data_source, pipeline, prefetch=False)
        self.aug = ScaleInstanceContrastAug(
            size=aug_dict['size'],
            scales=aug_dict['scales'],
            ratio=(3. / 4, 4. / 3),
            area_thr=aug_dict['area_thr'],
            part_thr=aug_dict['part_thr'],
            min_box_num=aug_dict['min_box_num'],
            max_try=aug_dict['max_try'],
        )
        self.max_box_num = data_source.max_box_num


    def __getitem__(self, idx):
        data_dict = self.data_source.get_sample(idx)

        img = Image.open(data_dict['file_name']).convert('RGB')

        img1, img2, boxes1, boxes2, classes = self.aug(img, data_dict['instances'], ret_class=True)

        n_boxes = len(boxes1)

        img1 = self.pipeline(img1)
        img2 = self.pipeline(img2)

        img_cat = torch.cat((img1.unsqueeze(0), img2.unsqueeze(0)), dim=0)

        boxes_ret1 = img_cat.new_full((self.max_box_num, 4), 0, dtype=torch.float32)
        boxes_ret2 = img_cat.new_full((self.max_box_num, 4), 0, dtype=torch.float32)

        boxes_ret1[:n_boxes] = boxes1
        boxes_ret2[:n_boxes] = boxes2

        classes_ret = img_cat.new_full((self.max_box_num,), -1, dtype=torch.int64)
        classes_ret[:n_boxes] = classes

        boxes_cat = torch.cat((boxes_ret1.unsqueeze(0), boxes_ret2.unsqueeze(0)), dim=0)
        boxes_num = img_cat.new_full((1,), 0, dtype=torch.int32) + n_boxes

        return dict(img=img_cat, boxes=boxes_cat, boxes_num=boxes_num, classes=classes_ret, img_id=int(data_dict['img_id']))


    def __len__(self):
        return self.data_source.get_length()

    def evaluate(self, scores, keyword, logger=None, **kwargs):
        raise NotImplemented