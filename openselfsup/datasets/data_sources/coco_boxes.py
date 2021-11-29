import os
import json
import torch
import numpy as np
from PIL import Image
import copy
import os
import logging

from detectron2.data import detection_utils as utils

from ..registry import DATASOURCES
from .load_coco import load_coco_json


@DATASOURCES.register_module
class COCO_BOXES(object):
    def __init__(self, root, json_file, max_box_num, image_format='RGB', *args, **kwargs):

        if json_file.endwith('instances_train2017.json'):
            logging.critical('Using ground-truth for pre-training, please use selective search result!')

        self.data_dicts = load_coco_json(json_file, root)
        self.image_format = image_format
        self.max_box_num = max_box_num

    def get_length(self):
        return len(self.data_dicts)

    def __len__(self):
        return self.get_length()

    def get_sample(self, idx):
        data_dict = self.data_dicts[idx]

        dataset_dict = copy.deepcopy(data_dict)  # it will be modified by code below
        annos = [obj for obj in dataset_dict.pop("annotations") if obj.get("iscrowd", 0) == 0]
        instances = utils.annotations_to_instances(annos, (dataset_dict['height'], dataset_dict['width']),)
        dataset_dict["instances"] = utils.filter_empty_instances(instances)

        return dataset_dict