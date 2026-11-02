"""Global Context Transformer (GCT) bottleneck.

Convolutional encoders have a limited effective receptive field, which hurts on large or
low-contrast lesions. The GCT flattens the lowest-resolution feature map into tokens, adds
fixed 2-D sine-cosine positional encodings and applies a stack of pre-norm Transformer
layers before projecting back and adding the result to the input (residual).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def sincos_2d(h: int, w: int, dim: int, device=None, dtype=None) -> torch.Tensor:
    """Fixed 2-D sine-cosine positional encoding of shape (h*w, dim)."""
    if dim % 4 != 0:
        raise ValueError("embedding dimension must be divisible by 4")
    y, x = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32),
        indexing="ij",
    )
    omega = torch.arange(dim // 4, device=device, dtype=torch.float32) / (dim // 4 - 1 if dim > 4 else 1)
    omega = 1.0 / (10000**omega)
    y = y.flatten()[:, None] * omega[None]
    x = x.flatten()[:, None] * omega[None]
    pe = torch.cat([x.sin(), x.cos(), y.sin(), y.cos()], dim=1)
    return pe.to(dtype) if dtype is not None else pe


class GlobalContextTransformer(nn.Module):
    def __init__(
        self,
        channels: int,
        dim: int = 256,
        depth: int = 4,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.proj_in = nn.Conv2d(channels, dim, 1)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=int(dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.proj_out = nn.Conv2d(dim, channels, 1)
        nn.init.zeros_(self.proj_out.weight)  # start as identity mapping (stable fine-tuning)
        nn.init.zeros_(self.proj_out.bias)
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        t = self.proj_in(x).flatten(2).transpose(1, 2)  # (B, HW, D)
        t = t + sincos_2d(h, w, self.dim, x.device, t.dtype)[None]
        t = self.norm(self.blocks(t))
        t = t.transpose(1, 2).reshape(b, self.dim, h, w)
        return x + self.proj_out(t)
