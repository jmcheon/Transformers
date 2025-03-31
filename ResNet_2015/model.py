from typing import List

import torch.nn as nn


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
        self.conv1 = nn.Conv2d(in_channels, f1, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(f1)

        # 2nd component
        self.conv2 = nn.Conv2d(f1, f2, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm2d(f2)

        # 3rd component
        self.conv3 = nn.Conv2d(f2, f3, kernel_size=1, stride=1, padding=0)
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
        self.conv1 = nn.Conv2d(in_channels, f1, kernel_size=1, stride=stride, padding=0)
        self.bn1 = nn.BatchNorm2d(f1)

        # 2nd component
        self.conv2 = nn.Conv2d(f1, f2, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm2d(f2)

        # 3rd component
        self.conv3 = nn.Conv2d(f2, f3, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(f3)

        # shortcut path
        self.shortcut_conv = nn.Conv2d(in_channels, f3, kernel_size=1, stride=stride, padding=0)
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


class ResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        """
        Stage-wise implementation of ResNet50 architecture:
        Conv2D -> BatchNorm -> ReLU -> MaxPool -> ConvBlock -> IdBlock * 2 -> ConvBlock -> IdBlock * 3
        -> ConvBlock -> IdBlock * 5 -> ConvBlock -> IdBlock * 2 -> AvgPool -> Flatten -> Dense
        """
        # stage 1

        # zero-padding
        self.pad = nn.ConstantPad2d(3, 0)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=0)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # stage 2
        self.stage2 = self._make_stage(
            ConvolutionalBlock,
            IdentityBlock,
            in_channels=64,
            filters=[64, 64, 256],
            blocks=3,
            stride=1,
        )

        # stage 3
        self.stage3 = self._make_stage(
            ConvolutionalBlock,
            IdentityBlock,
            in_channels=256,
            filters=[128, 128, 512],
            blocks=4,
            stride=2,
        )

        # stage 4
        self.stage4 = self._make_stage(
            ConvolutionalBlock,
            IdentityBlock,
            in_channels=512,
            filters=[256, 256, 1024],
            blocks=6,
            stride=2,
        )

        # stage 5
        self.stage5 = self._make_stage(
            ConvolutionalBlock,
            IdentityBlock,
            in_channels=1024,
            filters=[512, 512, 2048],
            blocks=2,
            stride=3,
        )

        # avg pool
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # output layer
        self.flatten = nn.Flatten()

    def _make_stage(self, conv_block, id_block, in_channels, filters, blocks, stride):
        layers = []

        layers.append(conv_block(in_channels, filters, kernel_size=3, stride=stride))

        for _ in range(1, blocks):
            layers.append(id_block(filters[2], filters, kernel_size=3))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)

        x = self.avgpool(x)
        x = self.flatten(x)

        return x


class ResNet50Classifier(nn.Moduel):
    def __init__(self, encoder: ResNet50, out_channels: int = 2048, num_classes: int = 6):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Linear(out_channels, num_classes)

    def forward(self, x):
        x = self.encoder(x)  # (B, out_channels)
        x = self.fc(x)  # (B, num_classes)

        return x
