import numpy as np
import torch
from utils.align import rand_offset, shift_hom

from transforms.functional import blosc_decode_tensor, blosc_encode_tensor
from transforms.registry import TRANSFORMS


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
class RandCropImgs:
    """ Random crop ab_T, fo_T, ab_R to the same size

        Assume input ab_T, fo_T have the same size"""

    def __init__(self, img_names=None, ratio=0.8, min_len=320, gcd=1, hom='hom_sift_tf') -> None:
        self.gcd = gcd
        self.ratio = ratio
        self.min_len = min_len
        self.img_names = img_names
        self.hom = hom

    def __call__(self, data):
        if self.img_names is None:
            self.img_names = tuple(data['imgs'].keys())
        shape_min = np.array(data['imgs'][self.img_names[0]].shape[-2:])

        shape_crop = np.minimum(shape_min, np.maximum(
            self.min_len, (shape_min * self.ratio).astype(np.int64)))

        if self.gcd != 1:
            shape_crop = shape_crop // self.gcd * self.gcd

        hc, wc = shape_crop

        it, jt = rand_offset(shape_min, shape_crop)
        if self.hom in data['hom']:
            data['hom'][self.hom] = shift_hom(
                data['hom'][self.hom], jt, it, jt, it)

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
class ToNumpy:
    def __call__(self, data):
        for k, v in data['imgs'].items():
            if len(v.shape) == 3:
                v = v.permute((1, 2, 0))

            data['imgs'][k] = v.numpy()
        return data


class BloscEncoder:
    def __init__(self, cname='lz4', clevel=3) -> None:
        self.cname = cname
        self.clevel = clevel

    def __call__(self, data):
        imgs = data['imgs']

        data['imgs'] = {k: blosc_encode_tensor(
            v, self.cname, self.clevel) for k, v in imgs.items()}
        return data


class BloscDecoder:
    def __init__(self, cname='lz4', clevel=3) -> None:
        pass

    def __call__(self, data):
        imgs = data['imgs']
        data['imgs'] = {k: blosc_decode_tensor(v) for k, v in imgs.items()}
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
