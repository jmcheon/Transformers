import torch.nn as nn
from model import (
    Encoder,
    EncoderBlock,
    FeedForwardBlock,
    MultiHeadAttentionBlock,
    PatchEmbedding,
)


class ViTModel(nn.Module):
    def __init__(self, patch_embed: PatchEmbedding, encoder: Encoder):
        super().__init__()
        self.patch_embed = patch_embed
        self.encoder = encoder

    def forward(self, x):
        # convert image to patch embeddings
        x = self.patch_embed(x)
        x = self.encoder(x)

        # extract CLS token (batch, d_model)
        cls_token_output = x[:, 0, :]

        return cls_token_output


class ViTClassifier(nn.Module):
    def __init__(self, vit: ViTModel, mlp_head: nn.Linear):
        super().__init__()
        self.vit = vit
        self.mlp_head = mlp_head

    def forward(self, x):
        cls_token_output = self.vit(x)
        logits = self.mlp_head(cls_token_output)

        return logits


def build_vitmodel(
    img_size: int,
    patch_size: int,
    in_channels: int,
    d_model: int,
    d_ff: int,
    h: int = 8,
    num_layers: int = 12,
    dropout: float = 0.1,
) -> ViTClassifier:
    patch_embed = PatchEmbedding(img_size, patch_size, in_channels, d_model)

    encoder_blocks = []
    for _ in range(num_layers):
        self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(self_attention_block, feed_forward_block, d_model, dropout)
        encoder_blocks.append(encoder_block)

    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))

    vit = ViTModel(patch_embed, encoder)

    for p in vit.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return vit


def build_vitclassifier(
    img_size: int,
    patch_size: int,
    in_channels: int,
    num_classes: int,
    d_model: int,
    d_ff: int,
    h: int = 8,
    num_layers: int = 12,
    dropout: float = 0.1,
) -> ViTClassifier:
    vit = build_vitmodel(img_size, patch_size, in_channels, d_model, d_ff, h, num_layers, dropout)

    mlp_head = nn.Linear(d_model, num_classes)

    classifier = ViTClassifier(vit, mlp_head)

    for p in classifier.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return classifier
