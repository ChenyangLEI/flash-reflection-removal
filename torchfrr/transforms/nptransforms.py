import numpy as np
from utils.align import rand_offset

from transforms.functional import blosc_decode_np, blosc_encode_np
from transforms.registry import TRANSFORMS


@TRANSFORMS.register
class RandCropThree:
    """ Random crop ab_T, fo, ab_R to the same size

        Assume input ab_T, fo have the same size"""

    def __init__(self, ratio=0.8, min_len=320, gcd=1) -> None:
        self.gcd = gcd
        self.ratio = ratio
        self.min_len = min_len

    def __call__(self, data):
        ab_T, ab_R = (data['imgs'][k] for k in ('ab_T', 'ab_R'))
        shape_T = np.array(ab_T.shape[:2])
        shape_R = np.array(ab_R.shape[:2])
        shape_min = np.minimum(shape_T, shape_R)

        shape_crop = np.minimum(shape_min, np.maximum(
            self.min_len, (shape_min * self.ratio).astype(np.int64)))

        if self.gcd != 1:
            shape_crop = shape_crop // self.gcd * self.gcd

        hc, wc = shape_crop

        it, jt = rand_offset(shape_T, shape_crop)
        ir, jr = rand_offset(shape_R, shape_crop)

        for name in ('ab_T', 'fo'):
            data['imgs'][name] = data['imgs'][name][it:it + hc, jt:jt + wc]

        data['imgs']['ab_R'] = data['imgs']['ab_R'][ir:ir + hc, jr:jr + wc]

        return data


class BloscEncoderNp:
    def __init__(self, cname='lz4', clevel=3) -> None:
        self.cname = cname
        self.clevel = clevel

    def __call__(self, data):
        imgs = data['imgs']

        data['imgs'] = {k: blosc_encode_np(
            v, self.cname, self.clevel) for k, v in imgs.items()}
        return data


class BloscDecoderNp:
    def __init__(self, cname='lz4', clevel=3) -> None:
        pass

    def __call__(self, data):
        imgs = data['imgs']
        data['imgs'] = {k: blosc_decode_np(v) for k, v in imgs.items()}
        return data
