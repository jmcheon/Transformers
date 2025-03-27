from typing import Tuple

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
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # conv layer to extract patches and project them to d_model
        self.proj = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Args:
            x: input image of size (batch, in_channels, img_size, img_size)

        Returns:
            tensor of shape (batch, num_patches, d_model)
        """
        # apply conv to get patch embeddings (batch, d_model, num_patches_h, num_patches_w)
        x = self.proj(x)

        # flatten the patches (batch, d_model, num_patches) -> (batch, num_patches, d_model)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim):
        super().__init__()
        self.input_resolution = input_resolution  # (H, W)
        self.dim = dim

        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x):
        """
        Args:
            x: (B, H*W, C) or (B, num_patches, d_model)
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "Input feature has wrong size"

        x = x.view(B, H, W, C)

        # split into 2x2 patches
        x0 = x[:, 0::2, 0::2, :]  # top-left
        x1 = x[:, 1::2, 0::2, :]  # bottom-left
        x2 = x[:, 0::2, 1::2, :]  # top-right
        x3 = x[:, 1::2, 1::2, :]  # bottom-right

        # concatenate along channel dimension
        x = torch.concat([x0, x1, x2, x3], dim=-1)  # (B, H/2, W/2, 4*C)
        # flatten spatial dimensions (B, new_num_patches, 4*C)
        x = x.view(B, -1, 4 * C)

        x = self.norm(x)
        # (B, new_num_patches, 4*C) -> (B, new_num_patches, 2*C)
        x = self.reduction(x)

        return x


class FeedForwardBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden_features = hidden_features or in_features * 4
        self.linear_1 = nn.Linear(in_features, hidden_features)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(hidden_features, out_features or in_features)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        x = self.dropout(x)

        return x


class WindowAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, window_size: int, dropout: float = 0.1):
        """
        Args:
            d_model (int): embedding dimension
            h (int): number of heads
            window_size (int): size of the attention window (7 x 7)
            dropout (float): dropout rate
        """
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.window_size = window_size
        assert d_model % h == 0, "d_model is not divisible by h"
        # scaled dot-product factor
        self.head_dim = d_model // self.h
        self.scale = (d_model // h) ** -5

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: intput tensor of shape (batch * num_window, window_size * window_size, d_model)

        Returns:
            output tensor of same shape
        """
        # (batch * num_windows, num_patches, d_model)
        batch_size, num_patches, d_model = x.shape

        query = self.query(x).view(batch_size, num_patches, self.h, self.head_dim).transpose(1, 2)
        key = self.key(x).view(batch_size, num_patches, self.h, self.head_dim).transpose(1, 2)
        value = self.value(x).view(batch_size, num_patches, self.h, self.head_dim).transpose(1, 2)

        # (batch, h, num_patches, head_dim) -> (batch, h, num_patches, num_patches)
        attention_scores = (query @ key.transpose(-2, -1)) * self.scale

        # mask (num_windows, window_size², window_size²)
        if mask is not None:
            attention_scores = attention_scores + mask

        attention_weights = attention_scores.softmax(dim=-1)
        attention_weights = self.dropout(attention_weights)

        x = attention_weights @ value

        x = x.transpose(1, 2).reshape(batch_size, num_patches, d_model)

        x = self.proj(x)

        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size: int

    Returns:
        windows: (B * num_windows, window_size * window_size, C)
    """
    B, H, W, C = x.shape
    # split into (B, num_windows_h, window_size, num_windows_w, window_size, C)
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)

    # permute to group patch tokens into windows
    # -> (B, num_windows_h, num_windows_w, window_size, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()

    # (B * num_windows, window_size * window_size, C)
    windows = x.view(-1, window_size * window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (B * num_windows, window_size * window_size, C)
        window_size: int
        H, W: height and width of original image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / ((H // window_size) * (W // window_size)))

    # (B, num_windows_h, num_windows_w, window_size, window_size, C)
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)

    # (B, num_windows_h, window_size, num_window_w, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, H, W, -1)

    return x


class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int],
        window_attention_block: WindowAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        window_size: int = 7,
        shift_size: int = 4,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution  # (H, W)
        self.window_size = window_size
        self.shift_size = shift_size

        assert 0 <= shift_size < window_size, "shift_size must be in [0, window_size)"

        self.norm1 = nn.LayerNorm(dim)
        self.window_attention_block = window_attention_block

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = feed_forward_block

    def create_attention_mask(self, H, W, device="cpu"):
        # region labels for each patch
        img_mask = torch.zeros((1, H, W, 1), device=device)

        cnt = 0

        for h in range(0, H, self.window_size):
            for w in range(0, W, self.window_size):
                img_mask[:, h : h + self.window_size, w : w + self.window_size, :] = cnt
                cnt += 1

        # apply cyclic shift
        shifted_mask = torch.roll(
            img_mask, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
        )

        # partition into windows
        B, H_, W_, C = shifted_mask.shape
        x = shifted_mask.view(
            B, H_ // self.window_size, self.window_size, W_ // self.window_size, self.window_size, C
        )
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()

        # (num_windows, window_size²)
        windows = x.view(-1, self.window_size * self.window_size)

        # compute the attention mask (num_windows, window_size², window_size²)
        attention_mask = windows.unsqueeze(1) - windows.unsqueeze(2)
        attention_mask = attention_mask.masked_fill(attention_mask != 0, float("-inf"))
        attention_mask = attention_mask.masked_fill(attention_mask == 0, float("0.0"))

        return attention_mask

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "Input has wrong spatial size"

        x = x.view(B, H, W, C)

        shortcut = x
        x = self.norm1(x)

        # cyclic shift if shift_size > 0
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows (num_window * B, window_size², C2)
        x_windows = window_partition(shifted_x, self.window_size)

        if self.shift_size > 0:
            attention_mask = self.create_attention_mask(H, W, device=x.device)
        else:
            attention_mask = None

        # self-attention & residual
        attention_window = self.window_attention_block(x_windows, attention_mask)

        # merge windows back (B, H, W, C)
        shifted_x = window_reverse(attention_window, self.window_size, H, W)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        # residual
        x = x + shortcut

        x = x.view(B, H * W, C)

        # FFN
        x = x + self.mlp(self.norm2(x))  # residual

        return x


class SwinEncoder(nn.Module):
    def __init__(self, layers=nn.ModuleList):
        super().__init__()
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
