import torch.nn as nn
from model import BasicBlock, ConvolutionalBlock, IdentityBlock


class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_dim = 512

        # stage 1
        self.pad = nn.ConstantPad2d(padding=3, value=0)  # match 7x7 kernel
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # stage 2
        self.stage2 = self._make_stage(64, 64, num_blocks=2, stride=1)
        self.stage3 = self._make_stage(64, 128, num_blocks=2, stride=2)
        self.stage4 = self._make_stage(128, 256, num_blocks=2, stride=2)
        self.stage5 = self._make_stage(256, self.output_dim, num_blocks=2, stride=2)

        # avg pool
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.flatten = nn.Flatten()

    def _make_stage(self, in_channels, out_channels, num_blocks, stride):
        layers = []

        # first block with downsampling
        downsampling = None
        if stride != 1 and in_channels != out_channels:
            downsampling = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers.append(BasicBlock(in_channels, out_channels, stride, downsampling))

        for _ in range(1, num_blocks):
            layers.append(BasicBlock(in_channels, out_channels))

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

        # (batch, 512)
        return x


class ResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        """
        Stage-wise implementation of ResNet50 architecture:
        Conv2D -> BatchNorm -> ReLU -> MaxPool -> ConvBlock -> IdBlock * 2 -> ConvBlock -> IdBlock * 3
        -> ConvBlock -> IdBlock * 5 -> ConvBlock -> IdBlock * 2 -> AvgPool -> Flatten
        """
        self.output_dim = 2048
        # stage 1

        # zero-padding
        self.pad = nn.ConstantPad2d(3, 0)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=0, bias=False)
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
            filters=[512, 512, self.output_dim],
            blocks=2,
            stride=3,
        )

        # avg pool
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # output layer
        self.flatten = nn.Flatten()

    def _make_stage(self, conv_block, id_block, in_channels, filters, num_blocks, stride):
        layers = []

        layers.append(conv_block(in_channels, filters, kernel_size=3, stride=stride))

        for _ in range(1, num_blocks):
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

        # (batch, 2048)
        return x


class ResNetClassifier(nn.Moduel):
    def __init__(self, encoder: nn.Module, num_classes: int = 6):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Linear(encoder.output_dim, num_classes)

    def forward(self, x):
        x = self.encoder(x)  # (batch, output_dim)
        x = self.fc(x)  # (batch, num_classes)

        return x
