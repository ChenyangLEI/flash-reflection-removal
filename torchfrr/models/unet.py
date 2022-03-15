import torch
from nn.conv import  UpCatDoubleConv, DownDoubleConv, UpCat
import torch.nn as nn
import torch.nn.functional as F
class UNet_SE(nn.Module):

    def __init__(self, in_channels=3, out_channels=3, hidden_channels=32, activation=nn.ReLU):
        super().__init__()
        self.input = nn.Sequential(nn.Conv2d(in_channels, hidden_channels, kernel_size=1, padding=0),
                                   activation(),
                                   nn.Conv2d(hidden_channels, hidden_channels,
                                             kernel_size=3, padding=1),
                                   activation())
        self.down1 = DownDoubleConv(
            hidden_channels, 2*hidden_channels, activation=activation)
        self.down2 = DownDoubleConv(
            2*hidden_channels, 4*hidden_channels, activation=activation)
        self.down3 = DownDoubleConv(
            4*hidden_channels, 8*hidden_channels, activation=activation)
        self.down4 = DownDoubleConv(
            8*hidden_channels, 16*hidden_channels, activation=activation)

        self.conv5= nn.Sequential(nn.Conv2d(16*hidden_channels, 16*hidden_channels,
                                             kernel_size=3, padding=1),
                                   activation())
        self.se=nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(-3,-1),
                nn.Linear(16*hidden_channels, 16*hidden_channels),
                                   activation(),
                                   nn.Sigmoid()
        )
        
        self.up1 = UpCatDoubleConv(
            16*hidden_channels, 8*hidden_channels, upconv=True, activation=activation)
        self.up2 = UpCatDoubleConv(
            8*hidden_channels, 4*hidden_channels, upconv=True, activation=activation)
        self.up3 = UpCatDoubleConv(
            4*hidden_channels, 2*hidden_channels, upconv=True, activation=activation)
        self.up4=UpCat(2*hidden_channels, hidden_channels, upconv=True)
        self.out = nn.Sequential(
                                 nn.Conv2d(2*hidden_channels, hidden_channels,
                                           kernel_size=3, padding=1),
                                 activation(),
                                 nn.Conv2d(hidden_channels, out_channels,
                                           kernel_size=3, padding=1),
                                 activation())

    def forward(self, x):
        x1 = self.input(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x5=self.conv5(x5)
        b,c=x5.shape[:2]
        se=self.se(x5).reshape(b,c,1,1)
        x5=se*x5

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.out(x)



class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=3, hidden_channels=32, activation=nn.ReLU):
        super().__init__()
        self.input = nn.Sequential(nn.Conv2d(in_channels, hidden_channels, kernel_size=1, padding=0),
                                   activation(),
                                   nn.Conv2d(hidden_channels, hidden_channels,
                                             kernel_size=3, padding=1),
                                   activation())
        self.down1 = DownDoubleConv(
            hidden_channels, 2*hidden_channels, activation=activation)
        self.down2 = DownDoubleConv(
            2*hidden_channels, 4*hidden_channels, activation=activation)
        self.down3 = DownDoubleConv(
            4*hidden_channels, 8*hidden_channels, activation=activation)
        self.down4 = DownDoubleConv(
            8*hidden_channels, 16*hidden_channels, activation=activation)
        self.up1 = UpCatDoubleConv(
            16*hidden_channels, 8*hidden_channels, upconv=True, activation=activation)
        self.up2 = UpCatDoubleConv(
            8*hidden_channels, 4*hidden_channels, upconv=True, activation=activation)
        self.up3 = UpCatDoubleConv(
            4*hidden_channels, 2*hidden_channels, upconv=True, activation=activation)
        self.up4=UpCat(2*hidden_channels, hidden_channels, upconv=True)
        self.out = nn.Sequential(
                                 nn.Conv2d(2*hidden_channels, hidden_channels,
                                           kernel_size=3, padding=1),
                                 activation(),
                                 nn.Conv2d(hidden_channels, out_channels,
                                           kernel_size=3, padding=1),
                                 activation())

    def forward(self, x):
        x1 = self.input(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.out(x)

