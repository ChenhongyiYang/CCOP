import sys
sys.path.append('.')
import copy
from detectron2.config import get_cfg

_base_ = '../../base.py'


# Detectron2 Backbone config
backbone_cfg = get_cfg()
backbone_cfg.MODEL.BACKBONE.FREEZE_AT = 0

backbone_cfg.MODEL.RESNETS.DEPTH = 50

backbone_cfg.MODEL.RESNETS.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
backbone_cfg.MODEL.RESNETS.NORM = 'BN'
backbone_cfg.MODEL.RESNETS.STRIDE_IN_1X1 = False


backbone_cfg.MODEL.FPN.IN_FEATURES = ["res2", "res3", "res4", "res5"]
backbone_cfg.MODEL.FPN.OUT_CHANNELS = 256
backbone_cfg.MODEL.FPN.NORM = 'BN'
backbone_cfg.MODEL.FPN.FUSE_TYPE = 'sum'

backbone_cfg.MODEL.ROI_HEADS.IN_FEATURES = ["p2", "p3", "p4", "p5"]
backbone_cfg.MODEL.ROI_HEADS.POOLER_SAMPLING_RATIO = 0

backbone_cfg.MODEL.PIXEL_MEAN = [123.675, 116.280, 103.530]
backbone_cfg.MODEL.PIXEL_STD = [58.395, 57.120, 57.375]

# model settings
model = dict(
    type='CCOP',
    backbone_cfg=backbone_cfg,
    pretrained=None,
    img_neck_params = dict(
        in_channels=2048,
        hid_channels=2048,
        embed_channels=128,
        with_avg_pool=True,
        num_mlp=1,
        norm='',
        last_bn=False),
    roi_neck_params = dict(
        in_channels=256,
        hid_channels=2048,
        embed_channels=128,
        with_avg_pool=False,
        num_mlp=1,
        norm='',
        last_bn=False),
    img_queue_len=65536,
    roi_queue_len=65536,
    momentum=0.999,
    img_temperature=0.2,
    roi_temperature=0.2,
    pooling_sizes=(7,),
    pooling_assign_sizes=((224, 192, 96, 48),),
    curr_params=dict(
        n_jitter=10,
        iou_thr=(0.8, 0.4),
        jit_mag=(0.15, 0.6, 0.35, 1.4)),
    total_step=800. * 462, # COCO train set, 256 batch size
)

# dataset settings
data_source_cfg = dict(
    type='COCO_BOXES',
    root='data/coco/train2017',
    json_file= 'data/coco/annotations_drops/train2017_ss_merged.json',
    image_format='RGB',
    max_box_num=100)

dataset_type = 'ContrastiveBox' 

train_pipeline1 = dict(
    ColorJitter=dict(
        brightness=0.4,
        contrast=0.4,
        saturation=0.2,
        hue=0.1,
        p=0.8),
    RandomGrayscale=dict(p=0.2),
    RandomGaussianBlur=dict(
        sigma=(0.1, 2.),
        kernel_size=(7, 7),
        p=0.5))

train_pipeline2 = copy.deepcopy(train_pipeline1)
model['aug_dicts'] = [train_pipeline1, train_pipeline2]

img_norm_cfg = dict(mean=backbone_cfg.MODEL.PIXEL_MEAN, std=backbone_cfg.MODEL.PIXEL_STD)
train_pipeline = [
    dict(type='ToTensor')
]

data = dict(
    imgs_per_gpu=32,
    workers_per_gpu=4,
    drop_last=True,
    train=dict(
        type=dataset_type,
        data_source=data_source_cfg,
        pipeline=train_pipeline,
        prefetch=False,
        aug_dict=dict(
            size=224,
            scales=[(0.1, 1.), (0.1, 1.)],
            area_thr=144,
            part_thr=0.6,
            min_box_num=1,
            max_try=10))
)

# optimizer
use_fp16 = False
optimizer_config = dict(use_fp16=use_fp16)
optimizer = dict(type='SGD', lr=0.3, weight_decay=0.0001, momentum=0.9)
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0.,
    warmup='linear',
    warmup_iters=10,
    warmup_ratio=0.0001, # cannot be 0
    warmup_by_epoch=True
)
checkpoint_config = dict(interval=50)
total_epochs = 800
