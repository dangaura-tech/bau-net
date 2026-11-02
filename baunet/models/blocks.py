"""Building blocks shared by the encoder and decoder."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """(Conv3x3 -> BN -> ReLU) x 2."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class ResidualBlock(nn.Module):
    """Pre-activation residual block with an optional 1x1 projection shortcut."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(self.drop(F.relu(self.bn2(out))))
        return out + self.shortcut(x)


class AttentionGate(nn.Module):
    """Additive attention gate (Oktay et al., 2018) applied to skip connections.

    Args:
        gate_ch: channels of the gating signal coming from the decoder.
        skip_ch: channels of the encoder skip feature map.
        inter_ch: channels of the intermediate projection.
    """

    def __init__(self, gate_ch: int, skip_ch: int, inter_ch: int) -> None:
        super().__init__()
        self.w_g = nn.Sequential(nn.Conv2d(gate_ch, inter_ch, 1, bias=False), nn.BatchNorm2d(inter_ch))
        self.w_x = nn.Sequential(nn.Conv2d(skip_ch, inter_ch, 1, bias=False), nn.BatchNorm2d(inter_ch))
        self.psi = nn.Sequential(nn.Conv2d(inter_ch, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if gate.shape[-2:] != skip.shape[-2:]:
            gate = F.interpolate(gate, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        alpha = self.psi(F.relu(self.w_g(gate) + self.w_x(skip)))
        return skip * alpha
