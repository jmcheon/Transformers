import math

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, img_size: int, patch_size: int, in_channels: int, d_model: int):
        """
        Args:
            img_size (int): image size
            patch_size (int): size of each patch
            in_channels (int): number of channels in the image
            d_model (int): dimension of the Transformer model embeddings
        """
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # conv layer to extract patches and project them to d_model
        self.proj = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        # self.proj = nn.Linear(in_channels * patch_size * patch_size, d_model)

        # learnable CLS token for classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # learnable positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, d_model))

    def forward(self, x):
        """
        Args:
            x: input image of size (batch, in_channels, img_size, img_size)

        Returns:
            tensor of shape (batch, num_patches + 1, d_model)
        """
        batch_size = x.shape[0]

        # apply conv to get patch embeddings (batch, d_model, num_patches_h, num_patches_w)
        x = self.proj(x)

        # flatten the patches (batch, d_model, num_patches) -> (batch, num_patches, d_model)
        x = x.flatten(2).transpose(1, 2)

        # add the CLS token to the beginning of the sequence
        cls_token = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, d_model)
        x = torch.cat((cls_token, x), dim=1)  # (batch, num_patches + 1, d_model)

        # add positional embeddings
        x = x + self.pos_embedding

        return x


class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 1e-5):
        """
        Args:
            eps (float): small constant for numerical stability
        """
        super().__init__()
        self.eps = eps
        self.gemma = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
        Args:
            x: input tensor of shape (batch_size, num_patches + 1, d_model)

        Returns:
            normalized tensor of same shape
        """
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gemma * (x - mean) / (std + self.eps) + self.beta


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        """
        Args:
            d_model (int): embedding dimension
            d_ff (int): inner layer dimension
            dropout (float): dropout rate
        """
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        """
        Args:
            x: input tensor of shape (batch_size, num_patches + 1, d_model)

        Returns:
            output tensor of same shape
        """
        # (batch, num_patches + 1, d_model) -> (batch, num_patches + 1, d_ff)
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.dropout(x)
        # (batch, num_patches + 1, d_ff) -> (batch, num_patches + 1, d_model)
        x = self.linear_2(x)

        return x


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float):
        """
        Args:
            d_model (int): embedding dimension
            h (int): number of heads
            dropout (float): dropout rate
        """
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h"

        # size per head
        self.head_dim = d_model // self.h

        # linear transformation for query, key, value
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # output projection
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: input tensor of shape (batch_size, num_patches + 1, d_model)

        Returns:
            output tensor of same shape
        """
        batch_size, num_patches, _ = x.shape

        # compute query, key, value matrices
        query = self.w_q(x)
        key = self.w_k(x)
        value = self.w_v(x)

        # (batch, num_patches + 1, d_model)
        # -> (batch, num_patches + 1, h, head_dim)
        # -> (batch, h, num_patches + 1, head_dim)
        query = query.view(batch_size, num_patches, self.h, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, num_patches, self.h, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, num_patches, self.h, self.head_dim).transpose(1, 2)

        # scaled dot-product attention
        # (batch, h, num_patches + 1, head_dim) -> (batch, h, num_patches + 1, num_patches + 1)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # softmax (batch, h, num_patches + 1. num_patches + 1)
        attention_weights = attention_scores.softmax(dim=-1)

        # dropout
        attention_weights = self.dropout(attention_weights)

        # weighted sum (batch, h, num_patches + 1, head_dim)
        x = attention_weights @ value

        # reshape back
        # (batch, h, num_patches + 1, head_dim)
        # -> (batch, num_patches + 1, h, head_dim)
        # -> (batch, num_patches + 1, d_model)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.head_dim)

        # output projection (batch, num_patches + 1, d_model) -> (batch, num_patches + 1, d_model)
        x = self.w_o(x)

        return x


class ResidualConnection(nn.Module):
    def __init__(self, dropout: float):
        """
        Args:
            dropout (float): dropout rate
        """
        super().__init__()
        self.norm = LayerNormalization()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        Args:
            x: input tensor of shape (batch, num_patches + 1, d_model)
            sublayer: the sublayer function to apply (attention or FFN)
        """
        # normalize x before sublayer
        # apply dropout after sublayer
        # add residual connection
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    def __init__(
        self,
        self_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
    ):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x):
        """
        Args:
            x: input tensor of shape (batch_size, num_patches + 1, d_model)

        Returns:
            output tensor of same shape
        """
        x = self.residual_connection[0](x, self.self_attention_block)
        x = self.residual_connection[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):
    def __init__(self, layers=nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x):
        """
        Args:
            x: input tensor of shape (batch_size, num_patches + 1, d_model)

        Returns:
            normalized output tensor of same shape.
        """
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
