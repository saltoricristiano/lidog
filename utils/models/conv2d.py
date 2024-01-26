""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, k=3, s=1, p=1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=k, padding=p, stride=s, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=k, padding=p, stride=s, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, k=3, p=1, s=1):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, k=k, p=p, s=s)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class DownNew(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, k=3, p=1, s=1):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            DoubleConv(in_channels, out_channels, k=k, p=p, s=s)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, k=3, p=1, s=1):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, k=k, p=p, s=s)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels // 2, out_channels, k=k, p=p, s=s)

    def forward(self, x1, hw):
        x1 = self.up(x1)
        # input is CHW
        diffY = hw[0] - x1.size()[2]
        diffX = hw[1] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        return self.conv(x1)


class UpOld(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, k=3, p=1, s=1):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, k=k, p=p, s=s)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, k=k, p=p, s=s)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class Decoder2D(nn.Module):
    def __init__(self, input_size, n_classes=7, bilinear=False, shapes=[[128, 128], [200, 200], [240, 240]]):
        super(Decoder2D, self).__init__()

        output_size = int(input_size / 2)
        self.up1 = Up(input_size, output_size, bilinear)
        input_size = int(input_size / 2)
        output_size = int(input_size / 2)
        self.up2 = Up(input_size, output_size, bilinear)
        input_size = int(input_size / 2)
        output_size = int(input_size / 2)
        self.up3 = Up(input_size, output_size, bilinear)

        self.out_conv = OutConv(output_size, n_classes)

        self.shapes = shapes

    def forward(self, x):
        x = self.up1(x, self.shapes[0])
        x = self.up2(x, self.shapes[1])
        x = self.up3(x, self.shapes[2])

        x = self.out_conv(x)
        return x


class Decoder2DExtended(nn.Module):
    def __init__(self, input_size, n_classes=7, bilinear=False, residual_dims=[128, 64, 32]):
        super(Decoder2DExtended, self).__init__()

        output_size = int(input_size / 2)
        self.up1 = Up(input_size, output_size, bilinear)
        input_size = int(input_size / 2)
        output_size = int(input_size / 2)
        self.up2 = Up(input_size + residual_dims[0], output_size, bilinear)
        input_size = int(input_size / 2)
        output_size = int(input_size / 2)
        self.up3 = Up(input_size+residual_dims[1], output_size, bilinear)
        input_size = int(input_size / 2)
        output_size = int(input_size / 2)
        self.up4 = Up(input_size+residual_dims[2], output_size, bilinear)

        self.out_conv = OutConv(output_size, n_classes)

    def forward(self, x):
        bev_feat_map_bottle, bev_feat_map_up1, bev_feat_map_up2, bev_feat_map_up3 = x
        x = self.up1(bev_feat_map_bottle, [64, 64])
        x = torch.cat([x, bev_feat_map_up1], dim=1)
        x = self.up2(x, [128, 128])
        x = torch.cat([x, bev_feat_map_up2], dim=1)
        x = self.up3(x, [200, 200])
        x = torch.cat([x, bev_feat_map_up3], dim=1)
        x = self.up4(x, [240, 240])

        x = self.out_conv(x)
        return x


class Encoder2D(nn.Module):
    def __init__(self, input_size, n_classes=7, binary_seg=False):
        super(Encoder2D, self).__init__()

        self.down1 = DownNew(input_size, 256, k=3, s=2, p=1)
        self.out_conv = OutConv(256, n_classes)
        self.binary_seg = binary_seg
        if self.binary_seg:
            self.binary_out_conv = OutConv(256, 2)

    def forward(self, x):
        x_down = self.down1(x)
        x = self.out_conv(x_down)
        if self.binary_seg:
            x_binary = self.binary_out_conv(x_down)
            return x, x_binary
        else:
            return x


class Encoder2DFC(nn.Module):
    def __init__(self, input_size, n_classes=7):
        super(Encoder2DFC, self).__init__()

        self.down1 = DownNew(input_size, 256, k=3, s=2, p=1)

        self.out_conv = OutConv(256, 256)

        self.fc = nn.Linear(256, n_classes)

    def forward(self, x):
        x = self.down1(x)
        x = self.out_conv(x)
        b, c, h, w = x.shape
        x = x.view(b, -1, c)
        xx = []
        for bb in range(b):
            xx.append(self.fc(x[bb]).view(1, -1, h, w))
        x = torch.cat(xx, dim=0)
        return x


class UNet2D(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet2D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64, k=3, s=1, p=1)
        self.down1 = Down(64, 128, k=3, s=1, p=1)
        self.down2 = Down(128, 256, k=3, s=1, p=1)
        self.down3 = Down(256, 512, k=3, s=1, p=1)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = UpOld(1024, 512 // factor, bilinear, k=3, s=1, p=1)
        self.up2 = UpOld(512, 256 // factor, bilinear, k=3, s=1, p=1)
        self.up3 = UpOld(256, 128 // factor, bilinear, k=3, s=1, p=1)
        self.up4 = UpOld(128, 64, bilinear, k=3, s=1, p=1)
        self.outc = OutConv(64, n_classes)

        # self.inc = DoubleConv(n_channels, 64, k=3, s=1, p=1)
        # self.down1 = Down(64, 128, k=3, s=1, p=1)
        # self.down2 = Down(128, 256, k=3, s=1, p=1)
        # self.down3 = Down(256, 512, k=3, s=1, p=1)
        # factor = 2 if bilinear else 1
        # self.down4 = Down(512, 1024 // factor)
        # self.up1 = UpOld(1024, 512 // factor, bilinear, k=3, s=1, p=1)
        # self.up2 = UpOld(512, 256 // factor, bilinear, k=3, s=1, p=1)
        # self.up3 = UpOld(256, 128 // factor, bilinear, k=3, s=1, p=1)
        # self.up4 = UpOld(128, 64, bilinear, k=3, s=1, p=1)
        # self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

