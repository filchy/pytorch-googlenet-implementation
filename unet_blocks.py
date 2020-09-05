import torch
import torch.nn as nn

class Double_Conv(nn.Module):
    """(2dConv -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super(Double_Conv, self).__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscale with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()

        self.max_pool_d_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            Double_Conv(in_channels=in_channels, out_channels=out_channels)
        )

    def forward(self, x):
        return self.max_pool_d_conv(x)


class Up(nn.Module):
    """Upscale then double conv"""
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        
        self.up = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=2, stride=2)

        self.double_conv = Double_Conv(in_channels=in_channels, out_channels=out_channels)

    @staticmethod
    def crop_map(x1, x2):
        """resize map x2 to targeted map x1"""
        target_size = x1.size()[2] # target_size < tensor_size
        tensor_size = x2.size()[2]

        delta = (tensor_size - target_size) // 2
        return x2[:, :, delta:tensor_size-delta, delta:tensor_size-delta]
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x2 = self.crop_map(x1, x2)

        x = torch.cat([x1, x2], dim=1)
        x = self.double_conv(x)
        return x

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.conv(x)
        return x1
