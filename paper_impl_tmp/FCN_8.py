import torch.nn as nn
from model import FCNDecoder, VGGEncoder


class SegmentationModel(nn.Module):
    def __init__(self, n_classes=12, pretrained=True):
        super().__init__()
        self.encoder = VGGEncoder(pretrained=pretrained)
        self.decoder = FCNDecoder(n_classes)

    def forward(self, x):
        encoder_outputs = self.encoder(x)

        logits = self.decoder(encoder_outputs)

        return logits
