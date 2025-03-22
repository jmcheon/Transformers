import torch.nn as nn
from model import (
    Encoder,
    EncoderBlock,
    FeedForwardBlock,
    MultiHeadAttentionBlock,
    PatchEmbedding,
)


class VisionTransformer(nn.Module):
    def __init__(self, patch_embed: PatchEmbedding, encoder: Encoder, mlp_head: nn.Linear):
        super().__init__()
        self.patch_embed = patch_embed
        self.encoder = encoder
        self.mlp_head = mlp_head

    def forward(self, x):
        # convert image to patch embeddings
        x = self.patch_embed(x)
        x = self.encoder(x)

        # extract CLS token (batch, d_model)
        cls_token_output = x[:, 0, :]
        logits = self.mlp_head(cls_token_output)

        return logits


def build_vision_transformer(
    img_size: int,
    patch_size: int,
    in_channels: int,
    num_classes: int,
    d_model: int,
    d_ff: int,
    h: int = 8,
    num_layers: int = 12,
    dropout: float = 0.1,
) -> VisionTransformer:
    patch_embed = PatchEmbedding(img_size, patch_size, in_channels, d_model)

    encoder_blocks = []
    for _ in range(num_layers):
        self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(self_attention_block, feed_forward_block, d_model, dropout)
        encoder_blocks.append(encoder_block)

    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))

    mlp_head = nn.Linear(d_model, num_classes)

    vit = VisionTransformer(patch_embed, encoder, mlp_head)

    for p in vit.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return vit
