import kornia
from kornia.constants import Resample
from kornia.color import *
from kornia import augmentation as K
import kornia.augmentation.random_generator as rg


import numpy as np
import torch
import torch.nn as nn

import torchvision.transforms as transforms


class AugmentationPipeline(nn.Module):
    def __init__(self, trans_dict):
        super(AugmentationPipeline, self).__init__()

        trans = []
        if 'ColorJitter' in trans_dict.keys():
            trans.append(
                K.ColorJitter(
                    brightness=trans_dict['ColorJitter']['brightness'],
                    contrast=trans_dict['ColorJitter']['contrast'],
                    saturation=trans_dict['ColorJitter']['saturation'],
                    hue=trans_dict['ColorJitter']['hue'],
                    p=trans_dict['ColorJitter']['p']
                )
            )

        if 'RandomGrayscale' in trans_dict.keys():
            trans.append(K.RandomGrayscale(p=trans_dict['RandomGrayscale']['p']))
        if 'RandomGaussianBlur' in trans_dict.keys():
            trans.append(
                K.RandomGaussianBlur(
                    sigma=trans_dict['RandomGaussianBlur']['sigma'],
                    kernel_size=trans_dict['RandomGaussianBlur']['kernel_size'],
                    p=trans_dict['RandomGaussianBlur']['p']
                )
            )
        if 'RandomHorizontalFlip' in trans_dict.keys():
            trans.append(K.RandomHorizontalFlip(p=trans_dict['RandomHorizontalFlip']['p']))
        if 'RandomSolarize' in trans_dict.keys():
            trans.append(
                K.RandomSolarize(
                    thresholds=trans_dict['RandomSolarize']['threshold'],
                    p=trans_dict['RandomSolarize']['p']
                ),
            )
        self.transform = transforms.Compose(trans)

    def forward(self, img):
        return self.transform(img)

