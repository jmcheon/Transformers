import torch.nn as nn


class SegNet(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        # encoder
        self.encoder = nn.ModuleList(
            [
                self._encoder_block(2, 3, 64),
                self._encoder_block(2, 64, 128),
                self._encoder_block(3, 128, 256),
                self._encoder_block(3, 256, 512),
                self._encoder_block(3, 512, 512),
            ]
        )

        # decoder (mirror of encoder)
        self.decoder = nn.ModuleList(
            [
                self._decoder_block(3, 512, 512),
                self._decoder_block(3, 512, 256),
                self._decoder_block(3, 256, 128),
                self._decoder_block(2, 128, 64),
                self._decoder_block(2, 64, num_classes, final=True),
            ]
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

    def _encoder_block(self, n_conv, in_channels, out_channels):
        layers = []
        for _ in range(n_conv):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            in_channels = out_channels
        layers.extend([nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)])

        return nn.Sequential(*layers)

    def _decoder_block(self, n_conv, in_channels, out_channels, final=False):
        layers = []

        for _ in range(n_conv):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            in_channels = out_channels
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        if final:
            layers.extend([nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)])

        return nn.Sequential(*layers)

    def forward(self, x):
        indices_lst = []
        sizes = []

        # encoder
        for block in self.encoder:
            x = block(x)
            sizes.append(x.size())
            x, indices = self.pool(x)
            indices_lst.append(indices)

        # decoder
        for i, block in enumerate(self.decoder):
            indices = indices_lst[-(i + 1)]
            size = sizes[-(i + 1)]
            x = self.unpool(x, indices, output_size=size)
            x = block(x)

        return x
