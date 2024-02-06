from functools import lru_cache

import lpips
import torch
import torch.nn.functional as F
import torchvision
from torch import nn

from transforms.registry import TRANSFORMS


@lru_cache
def get_perc_lossfn(num_layers=1, cuda=True):
    loss_fn = PerceptualLoss(num_layers=1)
    if cuda:
        loss_fn.cuda()
    return loss_fn

@ TRANSFORMS.register
class ImgsPerceptualLoss:
    def __init__(self, trips, num_layers=1, cuda=True):
        super().__init__()
        self.trips = trips
        self.loss_fn = get_perc_lossfn(num_layers, cuda)

    def __call__(self, data):
        for name, image_true, image_test in self.trips:
            data['losses'][name] = self.loss_fn(
                data['imgs'][image_true], data['imgs'][image_test])
        return data

@ TRANSFORMS.register
class ImgsLpips:
    def __init__(self, trips, num_layers=1, cuda=True):
        super().__init__()
        self.trips = trips
        self.loss_fn = lpips.LPIPS(net='alex')
        self.loss_fn.cuda()

    def __call__(self, data):
        for name, image_true, image_test in self.trips:
            data['metrics'][ 'lpips_' + name] = self.loss_fn(
                data['imgs'][image_true]*2-1, data['imgs'][image_test]*2-1)
        return data


class PerceptualLoss(nn.Module):
    def __init__(self, num_layers=1):
        super().__init__()
        self.register_buffer("mean", torch.tensor(
            [0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(
            [0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        vgg = torchvision.models.vgg19(pretrained=True)
        layers = [vgg.features[:4].eval(),
                  vgg.features[4:9].eval(),
                  vgg.features[9:14].eval(),
                  vgg.features[14:23].eval(),
                  vgg.features[23:32].eval(),
                  ][:num_layers]
        self.weights = [0.38461538, 0.20833333,
                        0.27027027, 0.17857143, 6.66666667]
        for l in layers:
            for p in l.parameters():
                p.requires_grad = False
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        x, y = [(t - self.mean) / self.std for t in (x, y)]
        loss = F.l1_loss(x, y)
        for i, layer in enumerate(self.layers):
            x, y = [layer(t) for t in (x, y)]
            loss += F.l1_loss(x, y) * self.weights[i]
        return loss
