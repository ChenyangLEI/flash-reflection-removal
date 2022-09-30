import torch

from transforms.functional import (peak_signal_noise_ratio,
                                   structural_similarity)
from transforms.registry import TRANSFORMS


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
class EndPointError:
    def __init__(self, trips, err_map=False):
        super().__init__()
        self.trips = trips
        self.err_map = err_map

    def __call__(self, data):
        for name, image_true, image_test in self.trips:
            err_map = torch.sqrt(torch.sum((data['imgs'][image_true] -
                                            data['imgs'][image_test])**2, dim=-3, keepdim=True))

            data['metrics']['epe_' + name] = err_map.mean()
            if self.err_map:
                data['imgs']['epe_' + name] = err_map
        return data
