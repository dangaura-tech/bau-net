"""Multi-scale encoders.

Every encoder returns a list of feature maps ordered from high to low resolution, starting
at full input resolution (stride 1) and ending at stride 2**(L-1). ``out_channels`` lists the
channel width of each level so that the decoder can be built for any backbone.
"""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import torchvision

from .blocks import ConvBlock, ResidualBlock


class ResidualEncoder(nn.Module):
    """Encoder trained from scratch: conv stem + max-pool/residual stages."""

    def __init__(self, in_channels: int, widths: Sequence[int], residual: bool = True) -> None:
        super().__init__()
        block = ResidualBlock if residual else ConvBlock
        self.stem = ConvBlock(in_channels, widths[0])
        self.stages = nn.ModuleList(
            [nn.Sequential(nn.MaxPool2d(2), block(widths[i], widths[i + 1])) for i in range(len(widths) - 1)]
        )
        self.out_channels = list(widths)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats = [self.stem(x)]
        for stage in self.stages:
            feats.append(stage(feats[-1]))
        return feats


class TorchvisionEncoder(nn.Module):
    """Wraps an ImageNet backbone (ResNet-34/50, ConvNeXt-T) as a 6-level pyramid (strides 1..32).

    A light full-resolution stem is added so the decoder can recover pixel-accurate
    boundaries; ConvNeXt (whose first stage is already stride 4) also gets a stride-2 stem.
    """

    SUPPORTED = ("resnet18", "resnet34", "resnet50", "convnext_tiny")

    def __init__(self, name: str, in_channels: int = 3, pretrained: bool = True, stem_width: int = 32) -> None:
        super().__init__()
        if name not in self.SUPPORTED:
            raise ValueError(f"encoder must be one of {self.SUPPORTED + ('residual',)}")
        if pretrained and in_channels != 3:
            raise ValueError("pretrained backbones expect 3 input channels")
        self.name = name
        net = torchvision.models.get_model(name, weights="DEFAULT" if pretrained else None)
        self.full_res = ConvBlock(in_channels, stem_width)
        if name.startswith("resnet"):
            if in_channels != 3:
                net.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
            self.stages = nn.ModuleList(
                [
                    nn.Sequential(net.conv1, net.bn1, net.relu),  # stride 2
                    nn.Sequential(net.maxpool, net.layer1),  # stride 4
                    net.layer2,  # stride 8
                    net.layer3,  # stride 16
                    net.layer4,  # stride 32
                ]
            )
            self._chained = True
        else:  # convnext: features = [stem(4x), stage1, down, stage2, down, stage3, down, stage4]
            f = net.features
            if in_channels != 3:
                f[0][0] = nn.Conv2d(in_channels, f[0][0].out_channels, 4, 4)
            self.half_res = nn.Sequential(
                nn.Conv2d(in_channels, 64, 3, 2, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True)
            )
            self.stages = nn.ModuleList(
                [
                    nn.Sequential(f[0], f[1]),
                    nn.Sequential(f[2], f[3]),
                    nn.Sequential(f[4], f[5]),
                    nn.Sequential(f[6], f[7]),
                ]
            )
            self._chained = False
        with torch.no_grad():
            was = self.training
            self.eval()
            dims = [t.shape[1] for t in self.forward(torch.zeros(1, in_channels, 64, 64))]
            self.train(was)
        self.out_channels = dims

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats = [self.full_res(x)]
        if self._chained:
            y = x
            for stage in self.stages:
                y = stage(y)
                feats.append(y)
        else:
            feats.append(self.half_res(x))
            y = x
            for stage in self.stages:
                y = stage(y)
                feats.append(y)
        return feats


def build_encoder(name: str, in_channels: int, widths: Sequence[int], residual: bool, pretrained: bool) -> nn.Module:
    if name == "residual":
        return ResidualEncoder(in_channels, widths, residual)
    return TorchvisionEncoder(name, in_channels, pretrained, stem_width=widths[0])
