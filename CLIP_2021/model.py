import torch.nn as nn
import torch.nn.functional as F
from Transformer_2017.model import (
    Encoder,
    EncoderBlock,
    FeedForwardBlock,
    InputEnbeddings,
    MultiHeadAttentionBlock,
    PositionalEncoding,
)


class CLIP(nn.Module):
    def __init__(self, image_encoder, text_encoder, embed_dim=512):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.image_proj = nn.Linear(image_encoder.output_dim, embed_dim)
        self.text_proj = nn.Linear(text_encoder.output_dim, embed_dim)

    def forward(self, image, text):
        image_features = self.image_encoder(image)
        text_features = self.text_encoder(text)

        image_embed = self.image_proj(image_features)
        text_embed = self.text_proj(text_features)

        image_embed = F.normalize(image_embed, dim=-1)
        text_embed = F.normalize(text_embed, dim=-1)

        return image_embed, text_embed


class CLIPTextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_len: int = 77,
        d_model: int = 512,
        d_ff: int = 2048,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = InputEnbeddings(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)

        layers = []

        for _ in range(num_layers):
            attn = MultiHeadAttentionBlock(d_model, num_heads, dropout)
            ffn = FeedForwardBlock(d_model, d_ff, dropout)
            block = EncoderBlock(attn, ffn, dropout)
            layers.append(block)

        self.encoder = Encoder(nn.ModuleList(layers))
        self.output_dim = d_model

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len), token IDs
        Returns:
            sentence embedding: (batch_size, d_model)
        """
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.encoder(x, mask)
        # mean pooling over tokens
        x = x.mean(dim=1)

        return x
