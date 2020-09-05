from unet_blocks import *

import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # input: 572x572 image
        self._d_conv_2d = Double_Conv(1, 64)
        self._down_1 = Down(64, 128)
        self._down_2 = Down(128, 256)
        self._down_3 = Down(256, 512)
        self._down_4 = Down(512, 1024)

        self._up_1 = Up(1024, 512)
        self._up_2 = Up(512, 256)
        self._up_3 = Up(256, 128)
        self._up_4 = Up(128, 64)

        self._out = OutConv(64, 2)

    def forward(self, x):
        # img encoder
        x1 = self._d_conv_2d(x)
        x2 = self._down_1(x1)
        x3 = self._down_2(x2)
        x4 = self._down_3(x3)
        x5 = self._down_4(x4)

        # img decoder
        x6 = self._up_1(x5, x4)
        x7 = self._up_2(x6, x3)
        x8 = self._up_3(x7, x2)
        x9 = self._up_4(x8, x1)

        x10 = self._out(x9)
        print(x10.size())

        return x10


if __name__ == "__main__":
    model = UNet().cuda()

    image = torch.rand((1, 1, 572, 572)).to(torch.device("cuda:0"))
    model(image)
