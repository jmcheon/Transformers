import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import VGG16_Weights, vgg16


class VGGBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        n_convs,
        kernel_size=3,
        activation=nn.ReLU,
        pool_size=2,
        pool_stride=2,
    ):
        super().__init__()

        layers = []

        for i in range(n_convs):
            layers.append(
                nn.Conv2d(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,  # same padding
                )
            )
            layers.append(activation(inplace=True))

        self.conv_layers = nn.Sequential(*layers)
        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_stride)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.pool(x)

        return x


class VGGEncoder(nn.Module):
    def __init__(self, input_channels=3, pretrained=True):
        super().__init__()

        self.block1 = VGGBlock(input_channels, 64, n_convs=2)
        self.block2 = VGGBlock(64, 128, n_convs=2)
        self.block3 = VGGBlock(128, 256, n_convs=3)
        self.block4 = VGGBlock(256, 512, n_convs=3)
        self.block5 = VGGBlock(512, 512, n_convs=3)

        self.conv6 = nn.Conv2d(512, 4096, kernel_size=7, padding=3)  # same padding
        self.relu6 = nn.ReLU(inplace=True)

        self.conv7 = nn.Conv2d(4096, 4096, kernel_size=1)  # 1x1 conv
        self.relu7 = nn.ReLU(inplace=True)

        if pretrained:
            self._load_pretrained_weights()

    def forward(self, x):
        p1 = self.block1(x)  # (B,64,H/2,W/2)
        p2 = self.block2(p1)  # (B,128,H/4,W/4)
        p3 = self.block3(p2)  # (B,256,H/8,W/8)
        p4 = self.block4(p3)  # (B,512,H/16,W/16)
        p5 = self.block5(p4)  # (B,512,H/32,W/32)

        c6 = self.relu6(self.conv6(p5))  # (B,4096,H/32,W/32)
        c7 = self.relu7(self.conv7(c6))  # (B,4096,H/32,W/32)

        return p1, p2, p3, p4, c7

    def _load_pretrained_weights(self):
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        pretrained_dict = vgg.features.state_dict()

        model_dict = self.state_dict()

        # filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict, strict=False)
        print("Pretrained VGG16 weights loaded successfully.")


class FCNDecoder(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

        # 1x1 convolutions to reduce depth of features from encoder
        self.conv1x1_f4 = nn.Conv2d(512, n_classes, kernel_size=1)
        self.conv1x1_f3 = nn.Conv2d(256, n_classes, kernel_size=1)

        # transposed convolutions (upsampling)
        self.upconv_f5 = nn.ConvTranspose2d(
            n_classes, n_classes, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.upconv_f4 = nn.ConvTranspose2d(
            n_classes, n_classes, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.upconv_final = nn.ConvTranspose2d(
            n_classes, n_classes, kernel_size=16, stride=8, padding=4, bias=False
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, encoder_outputs):
        # unpack encoder outputs
        f1, f2, f3, f4, f5 = encoder_outputs

        # 1st upsample by factor of 2 (f5 -> f4 resolution)
        o = self.upconv_f5(f5)  # (H/16, W/16)
        o2 = self.conv1x1_f4(f4)
        o2 = self.relu(o2)
        o = o + o2  # add skip connection

        # 2nd upsampling step
        o = self.upconv_f4(o)  # (H/8, W/8)
        o2 = self.conv1x1_f3(f3)  # skip connection from f3
        o2 = self.relu(o2)
        o = o + o2

        # final upsampling to original image resolution
        # final output: (B, n_classes, H, W)
        o = self.upconv_final(o)

        o = F.softmax(o, dim=1)
        return o
