from typing import List

import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsampling=None):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # shortcut projection
        self.downsmapling = downsampling

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsmapling is not None:
            identity = self.downsmapling(identity)

        x += identity
        x = self.relu(x)

        return x


class IdentityBlock(nn.Module):
    def __init__(self, in_channels: int, filters: List[int], kernel_size: int):
        super().__init__()
        """
        Identity block that skips over 3 layers

        Args:
            in_channels (int): number of input channels
            filters (List[int]): number of filters
            kernel_size (int): the shape of the middle Conv's window for the main path
        """

        # retrieve filters
        f1, f2, f3 = filters

        # 1st component of the main path
        self.conv1 = nn.Conv2d(in_channels, f1, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(f1)

        # 2nd component
        self.conv2 = nn.Conv2d(
            f1, f2, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=False
        )
        self.bn2 = nn.BatchNorm2d(f2)

        # 3rd component
        self.conv3 = nn.Conv2d(f2, f3, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(f3)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # save input value
        input_shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        # final step: add shortcut value to main path, and pass it through a ReLU
        x += input_shortcut
        x = self.relu(x)

        return x


class ConvolutionalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        filters: List[int],
        kernel_size: int,
        stride: int = 2,
    ):
        """
        Convolutional block that skips over 3 layers

        Args:
            in_channels (int): number of input channels
            filters (List[int]): number of filters
            kernel_size (int): the shape of the middle Conv's window for the main path
            stride (int): stride value
        """

        # retrieve filters
        f1, f2, f3 = filters

        # 1st component of main path
        self.conv1 = nn.Conv2d(in_channels, f1, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(f1)

        # 2nd component
        self.conv2 = nn.Conv2d(
            f1, f2, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=False
        )
        self.bn2 = nn.BatchNorm2d(f2)

        # 3rd component
        self.conv3 = nn.Conv2d(f2, f3, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(f3)

        # shortcut path
        self.shortcut_conv = nn.Conv2d(
            in_channels, f3, kernel_size=1, stride=stride, padding=0, bias=False
        )
        self.shortcut_bn = nn.BatchNorm2d(f3)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # save input value
        input_shortcut = self.shortcut_conv(x)
        input_shortcut = self.shortcut_bn(input_shortcut)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        # final step: add shortcut value to main path, and pass it through a ReLU
        x += input_shortcut
        x = self.relu(x)

        return x
