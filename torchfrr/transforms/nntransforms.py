from itertools import accumulate

import torch
from nn.conv import DownDoubleConv, UpCat, UpCatDoubleConv
from nn.loss import get_perc_lossfn
from torch import nn

from transforms.registry import TRANSFORMS


@TRANSFORMS.register
class MultiUNet(nn.Module):
    def __init__(self, io_ls=(
        ((('ab', 3), ('fo', 3)), (('ab_R_pred', 3),)),
        ((('ab', 3), ('ab_R_pred', 3)), (('ab_T_pred', 3),))
    ), *args, **kwargs):
        super().__init__()
        self.layers = nn.Sequential(*[UNet(in_ls, out_ls, *args, **kwargs)
                                      for in_ls, out_ls in io_ls])

    def forward(self, data):
        """ab, fo: B, C, H, W"""

        return self.layers(data)


@TRANSFORMS.register
class UNet(nn.Module):
    def __init__(self,
                 in_ls=(('ab', 3), ('fo', 3)), out_ls=(('ab_R_pred', 3),),
                 activation="LeakyReLU", se=True, hidden_channels=32):
        super().__init__()
        activation = getattr(nn, activation)
        self.in_names = [x[0] for x in in_ls]
        self.out_names = [x[0] for x in out_ls]
        self.out_endpoints = [0] + list(accumulate(x[1] for x in out_ls))
        self.input = nn.Sequential(nn.Conv2d(sum(x[1] for x in in_ls), hidden_channels, kernel_size=1, padding=0),
                                   activation(),
                                   nn.Conv2d(hidden_channels, hidden_channels,
                                             kernel_size=3, padding=1),
                                   activation())
        self.down1 = DownDoubleConv(
            hidden_channels, 2 * hidden_channels, activation=activation)
        self.down2 = DownDoubleConv(
            2 * hidden_channels, 4 * hidden_channels, activation=activation)
        self.down3 = DownDoubleConv(
            4 * hidden_channels, 8 * hidden_channels, activation=activation)
        self.down4 = DownDoubleConv(
            8 * hidden_channels, 16 * hidden_channels, activation=activation)
        if se:
            self.conv5 = nn.Sequential(nn.Conv2d(16 * hidden_channels, 16 * hidden_channels,
                                                 kernel_size=3, padding=1),
                                       activation())
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(-3, -1),
                nn.Linear(16 * hidden_channels, 16 * hidden_channels),
                activation(),
                nn.Sigmoid()
            )
        else:
            self.se = None
        self.up1 = UpCatDoubleConv(
            16 * hidden_channels, 8 * hidden_channels, upconv=True, activation=activation)
        self.up2 = UpCatDoubleConv(
            8 * hidden_channels, 4 * hidden_channels, upconv=True, activation=activation)
        self.up3 = UpCatDoubleConv(
            4 * hidden_channels, 2 * hidden_channels, upconv=True, activation=activation)
        self.up4 = UpCat(2 * hidden_channels, hidden_channels, upconv=True)
        self.out = nn.Sequential(
            nn.Conv2d(2 * hidden_channels, hidden_channels,
                      kernel_size=3, padding=1),
            activation(),
            nn.Conv2d(hidden_channels, sum(x[1] for x in out_ls),
                      kernel_size=3, padding=1),
            activation())

    def forward(self, data):
        x = torch.cat([data['imgs'][name]
                      for name in self.in_names], dim=-3)
        x1 = self.input(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        if self.se is not None:
            x5 = self.conv5(x5)
            b, c = x5.shape[:2]
            se = self.se(x5).reshape(b, c, 1, 1)
            x5 = se * x5
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.out(x)

        endpoints = self.out_endpoints
        for i, name in enumerate(self.out_names):
            data['imgs'][name] = out[..., endpoints[i]:endpoints[i + 1], :, :]

        return data

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

