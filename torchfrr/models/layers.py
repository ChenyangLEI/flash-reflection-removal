import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, activation=nn.ReLU):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3,  padding=1),
            activation(),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1),
            activation()
        )

    def forward(self, x):
        return self.layers(x)


class DownDoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, activation=nn.ReLU):
        super().__init__()
        self.layers = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, activation=activation)
        )

    def forward(self, x):
        return self.layers(x)


class UpCat(nn.Module):
    def __init__(self, in_channels, out_channels, upconv=False):
        super().__init__()
        layers = [nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)]
        if upconv:
            layers.append(
                nn.Conv2d(in_channels, out_channels, 3, padding=1))
        self.up = nn.Sequential(*layers)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        dh = x2.shape[-2]-x1.shape[-2]
        dw = x2.shape[-1]-x1.shape[-1]
        x1 = F.pad(x1, [dw//2, dw-dw//2, dh//2, dh-dh//2])

        return torch.cat([x1, x2], dim=1)


class UpCatDoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, upconv=False, activation=nn.ReLU):
        super().__init__()
        self.upcat = UpCat(in_channels, out_channels, upconv)
        self.double_conv = DoubleConv(
            2*out_channels, out_channels, activation=activation)

    def forward(self, x1, x2):
        return self.double_conv(self.upcat(x1, x2))
