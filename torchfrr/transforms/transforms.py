from functools import partial

import numpy as np
import torch
import cv2 as cv

from transforms.registry import TRANSFORMS
from transforms.functional import peak_signal_noise_ratio, structural_similarity


@TRANSFORMS.register
class LossesWeightSum:
    def __init__(self, pairs):
        super().__init__()
        self.pairs = pairs

    def __call__(self, data):
        total_loss = 0
        total_weight = 0
        for name, weight in self.pairs:
            total_loss += data['losses'][name] * weight
            total_weight += weight
        data['losses']['total'] = total_loss / total_weight

        return data


@TRANSFORMS.register
class ImgsSsim:
    def __init__(self, trips):
        super().__init__()
        self.trips = trips

    def __call__(self, data):
        for name, image_true, image_test in self.trips:
            data['metrics']['ssim_' + name] = structural_similarity(
                data['imgs'][image_true], data['imgs'][image_test])
        return data


@TRANSFORMS.register
class ImgsPsnr:
    def __init__(self, trips, err_map=False):
        super().__init__()
        self.trips = trips
        self.err_map = err_map

    def __call__(self, data):
        for name, image_true, image_test in self.trips:
            psnr, err = peak_signal_noise_ratio(
                data['imgs'][image_true], data['imgs'][image_test])
            data['metrics']['psnr_' + name] = psnr
            if self.err_map:
                data['imgs']['err_' + name] = err.mean(dim=-3, keepdims=True)
        return data


@TRANSFORMS.register
class GammaCorrection:
    def __init__(self, img_names=None, gamma=1 / 2.2) -> None:
        self.img_names = img_names
        self.gamma = gamma

    def __call__(self, data):
        if self.img_names is None:
            self.img_names = tuple(data['imgs'].keys())
        for img_name in self.img_names:
            data['imgs'][img_name] = data['imgs'][img_name]**self.gamma

        return data


@TRANSFORMS.register
class InverseGamma(GammaCorrection):
    def __init__(self, img_names, gamma=1 / 2.2) -> None:
        super().__init__(img_names, 1 / gamma)


class FilterImgs:
    def __init__(self, filters) -> None:
        self.filters = filters

    def __call__(self, data):
        data['imgs'] = {k: data['imgs'][k] for k in self.filters}

        return data


@TRANSFORMS.register
class MeanImgs:
    def __init__(self, img_names) -> None:
        self.img_names = img_names
        self.fn = None

    def __call__(self, data):
        if self.fn is None:
            if isinstance(next(iter(data['imgs'].values())), np.ndarray):
                self.fn = partial(np.mean, axis=-1, keepdims=True)
            else:
                self.fn = partial(torch.mean, dim=-3, keepdims=True)
        for k in self.img_names:
            data['imgs'][k] = self.fn(data['imgs'][k])

        return data


@TRANSFORMS.register
class DimImgs:
    def __init__(self, pairs) -> None:
        self.pairs = pairs

    def __call__(self, data):
        for name, ratio in self.pairs:
            if data['imgs'][name].max() > ratio:
                data['imgs'][name] *= ratio

        return data


@TRANSFORMS.register
class ImgCmds:
    def __init__(self, cmds) -> None:
        self.cmds = compile('\n'.join(cmds), 'ImgCmds', 'exec')

    def __call__(self, data):
        exec(self.cmds, globals(), data['imgs'])

        return data


@TRANSFORMS.register
class AddImgs:
    def __init__(self, trips) -> None:
        self.trips = trips

    def __call__(self, data):
        for a, b, s in self.trips:
            data['imgs'][s] = data['imgs'][a] + data['imgs'][b]

        return data


@TRANSFORMS.register
class AssignImgs:
    def __init__(self, pairs) -> None:
        self.pairs = pairs

    def __call__(self, data):
        for d in ['img_names', 'imgs']:
            for src, dst in self.pairs:
                data[d][dst] = data[d][src]

        return data


@ TRANSFORMS.register
class RenameImgs:
    def __init__(self, pairs) -> None:
        self.pairs = pairs

    def __call__(self, data):
        for src, dst in self.pairs:
            data['imgs'][dst] = data['imgs'].pop(src)

        return data
