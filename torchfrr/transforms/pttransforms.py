

import numpy as np
import torch

from transforms.registry import TRANSFORMS
from utils.align import rand_offset

@TRANSFORMS.register
class ToCuda:
    def __init__(self, img_names=['ab_R', 'ab', 'ab_T', 'fo', 'flow_tf']) -> None:
        self.img_names = img_names

    def __call__(self, data):
        for n in self.img_names:
            if n in data['imgs']:
                data['imgs'][n] = data['imgs'][n].cuda()
        return data



@ TRANSFORMS.register
class CropBigImgs:
    """ Random crop ab_T, fo_T, ab_R to the same size

        Assume input ab_T, fo_T have the same size"""

    def __init__(self,random=False,max_pix=640000, img_names=None, crop_ratio=3/4, gcd=32) -> None:
        self.gcd = gcd
        self.max_pix=max_pix
        self.img_names = img_names
        self.crop_ratio=crop_ratio
        self.random=random

    def __call__(self, data):
        if self.img_names is None:
            self.img_names = tuple(data['imgs'].keys())
        shape_origin = np.array(data['imgs'][self.img_names[0]].shape[-2:])
        shape_crop=shape_origin//self.gcd *self.gcd
        if shape_crop[0]*shape_crop[1]>self.max_pix:
            shape_crop=(shape_crop* self.crop_ratio).astype(np.int64) // self.gcd * self.gcd

        hc, wc = shape_crop

        if self.random:
            it, jt = rand_offset(shape_origin, shape_crop)
        else:
            it, jt = (shape_origin-shape_crop)//2


        for name in self.img_names:
            data['imgs'][name] = data['imgs'][name][..., it:it + hc, jt:jt + wc]

        return data



@ TRANSFORMS.register
class ToTensor:
    def __call__(self, data):
        for k, v in data['imgs'].items():
            if len(v.shape) == 3:
                v = v.transpose((2, 0, 1))
            else:
                v = v[np.newaxis, :, :]
            data['imgs'][k] = torch.from_numpy(
                v.astype(np.float32)).contiguous()
        return data


@TRANSFORMS.register
class ClampImgs:
    def __init__(self, img_names=None, minimum=0, maximum=1) -> None:
        self.min = minimum
        self.max = maximum
        self.img_names = img_names

    def __call__(self, data):
        if self.img_names is None:
            self.img_names = tuple(data['imgs'].keys())
        for img_name in self.img_names:
            if img_name in data['imgs']:
                data['imgs'][img_name] = torch.clamp(
                    data['imgs'][img_name], self.min, self.max)

        return data

