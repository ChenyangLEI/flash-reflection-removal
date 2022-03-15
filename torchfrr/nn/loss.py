import enum
import os
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from functools import lru_cache


@lru_cache
def get_perc_lossfn(num_layers=1, cuda=True):
    loss_fn = PerceptualLoss(num_layers=1)
    if cuda:
        loss_fn.cuda()
    return loss_fn


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
