from typing import List

import torch.nn as nn
from model import (
    FeedForwardBlock,
    PatchEmbedding,
    PatchMerging,
    SwinEncoder,
    SwinTransformerBlock,
    WindowAttentionBlock,
)


class SwinStage(nn.Module):
    def __init__(
        self,
        dim: int,
        input_resolution: int,
        depth: int,
        num_heads: int,
        window_size: int,
        downsample: PatchMerging,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()

        for i in range(depth):
            shift_size = 0 if i % 2 == 0 else window_size // 2
            window_attention_block = WindowAttentionBlock(dim, num_heads, window_size, dropout)
            feed_forward_block = FeedForwardBlock(dim, int(dim * mlp_ratio), dropout=dropout)
            swin_block = SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                window_attention_block=window_attention_block,
                feed_forward_block=feed_forward_block,
                window_size=window_size,
                shift_size=shift_size,
            )
            self.blocks.append(swin_block)

        self.downsample = downsample

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        if self.downsample is not None:
            x = self.downsample(x)

        return x


class SwinModel(nn.Module):
    def __init__(
        self,
        num_features: int,
        patch_embed: PatchEmbedding,
        encoder: SwinEncoder,
    ):
        super().__init__()
        self.patch_embed = patch_embed
        self.encoder = encoder
        self.layernorm = nn.LayerNorm(num_features)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.encoder(x)

        x = self.layernorm(x)
        # global avg pool
        x = x.mean(dim=1)

        return x


class SwinClassifier(nn.Module):
    def __init__(self, swin: SwinModel, classifier: nn.Linear):
        super().__init__()
        self.swin = swin
        self.classifier = classifier

    def forward(self, x):
        x = self.swin(x)
        logits = self.classifier(x)

        return logits


def build_swin_classifier(
    img_size: int = 224,
    patch_size: int = 4,
    in_channels: int = 4,
    num_classes: int = 1000,
    d_model: int = 96,
    num_heads: List[int] = [3, 6, 12, 24],
    depths: List[int] = [2, 2, 6, 2],
    window_size: int = 7,
) -> SwinClassifier:
    patch_embed = PatchEmbedding(img_size, patch_size, in_channels, d_model)

    patch_resolution = (img_size // patch_size, img_size // patch_size)
    num_layers = len(depths)
    num_features = int(d_model * 2 ** (num_layers - 1))

    stages = []

    dim = d_model
    resolution = patch_resolution

    for i in range(num_layers):

        if i < num_layers - 1:
            downsample = PatchMerging(input_resolution=resolution, dim=dim)
        else:
            downsample = None

        stage = SwinStage(
            dim=dim,
            input_resolution=resolution,
            depth=depths[i],
            num_heads=num_heads[i],
            window_size=window_size,
            downsample=downsample,
        )
        stages.append(stage)

        if i < num_layers - 1:
            dim *= 2
            resolution = (resolution[0] // 2, resolution[1] // 2)


    swin_encoder = SwinEncoder(nn.ModuleList(stages))

    swin = SwinModel(num_features, patch_embed, swin_encoder)

    classifier = nn.Linear(num_features, num_classes)

    swin_classifier = SwinClassifier(swin, classifier)

    for p in swin_classifier.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return swin_classifier
