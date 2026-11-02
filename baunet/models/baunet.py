"""BAU-Net architecture.

The network is a residual U-Net with attention-gated skip connections and two output
heads that share the decoder:

* a **mask head** predicting the foreground probability, and
* a **boundary head** predicting a thin band around the object contour.

The boundary prediction is fused back into the mask head (boundary-to-mask fusion), which
encourages sharper contours. Spatial dropout in the decoder enables Monte-Carlo dropout
at test time for pixel-wise epistemic uncertainty.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import AttentionGate, ConvBlock, ResidualBlock


class Encoder(nn.Module):
    def __init__(self, in_ch: int, widths: Sequence[int], residual: bool) -> None:
        super().__init__()
        block = ResidualBlock if residual else ConvBlock
        self.stem = ConvBlock(in_ch, widths[0])
        self.stages = nn.ModuleList(
            [nn.Sequential(nn.MaxPool2d(2), block(widths[i], widths[i + 1])) for i in range(len(widths) - 1)]
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats = [self.stem(x)]
        for stage in self.stages:
            feats.append(stage(feats[-1]))
        return feats


class DecoderStage(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, attention: bool, dropout: float, residual: bool) -> None:
        super().__init__()
        block = ResidualBlock if residual else ConvBlock
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.gate = AttentionGate(out_ch, skip_ch, max(skip_ch // 2, 1)) if attention else None
        self.block = block(out_ch + skip_ch, out_ch, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        if self.gate is not None:
            skip = self.gate(x, skip)
        return self.block(torch.cat([x, skip], dim=1))


class BAUNet(nn.Module):
    """Boundary-Aware U-Net.

    Args:
        in_channels: number of input image channels (3 for RGB endoscopy / dermoscopy).
        num_classes: number of foreground classes (1 for binary segmentation).
        widths: channel widths per resolution level, from high to low resolution.
        attention: use attention gates on the skip connections.
        boundary_head: add the auxiliary boundary head and boundary-to-mask fusion.
        residual: use residual blocks in encoder/decoder instead of plain conv blocks.
        dropout: spatial dropout probability in the decoder (enables MC-dropout).
        deep_supervision: also return low-resolution auxiliary mask logits.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        widths: Sequence[int] = (32, 64, 128, 256, 512),
        attention: bool = True,
        boundary_head: bool = True,
        residual: bool = True,
        dropout: float = 0.1,
        deep_supervision: bool = False,
    ) -> None:
        super().__init__()
        widths = list(widths)
        self.num_classes = num_classes
        self.use_boundary = boundary_head
        self.deep_supervision = deep_supervision
        self.encoder = Encoder(in_channels, widths, residual)
        rev = widths[::-1]
        self.decoder = nn.ModuleList(
            [DecoderStage(rev[i], rev[i + 1], rev[i + 1], attention, dropout, residual) for i in range(len(rev) - 1)]
        )
        head_ch = widths[0]
        if boundary_head:
            self.boundary_head = nn.Sequential(
                nn.Conv2d(head_ch, head_ch // 2, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(head_ch // 2, 1, 1)
            )
            self.fuse = nn.Conv2d(head_ch + 1, head_ch, 3, padding=1)
        self.mask_head = nn.Conv2d(head_ch, num_classes, 1)
        if deep_supervision:
            self.aux_heads = nn.ModuleList([nn.Conv2d(c, num_classes, 1) for c in rev[1:-1]])
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats = self.encoder(x)
        y = feats[-1]
        skips = feats[:-1][::-1]
        aux = []
        for i, (stage, skip) in enumerate(zip(self.decoder, skips, strict=True)):
            y = stage(y, skip)
            if self.deep_supervision and i < len(self.decoder) - 1:
                aux.append(self.aux_heads[i](y))
        out: Dict[str, torch.Tensor] = {}
        if self.use_boundary:
            boundary = self.boundary_head(y)
            y = F.relu(self.fuse(torch.cat([y, torch.sigmoid(boundary)], dim=1))) + y
            out["boundary"] = boundary
        out["mask"] = self.mask_head(y)
        if aux:
            out["aux"] = [F.interpolate(a, size=x.shape[-2:], mode="bilinear", align_corners=False) for a in aux]
        return out

    @torch.no_grad()
    def predict_with_uncertainty(self, x: torch.Tensor, n_samples: int = 10) -> Dict[str, torch.Tensor]:
        """Monte-Carlo dropout inference.

        Returns the mean foreground probability, the predictive entropy (total uncertainty)
        and the mutual information (epistemic uncertainty) per pixel.
        """
        was_training = self.training
        self.eval()
        for m in self.modules():
            if isinstance(m, nn.Dropout2d):
                m.train()
        probs = torch.stack([torch.sigmoid(self(x)["mask"]) for _ in range(n_samples)])
        self.train(was_training)
        mean = probs.mean(0)
        eps = 1e-8

        def entropy(p: torch.Tensor) -> torch.Tensor:
            return -(p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps))

        total = entropy(mean)
        aleatoric = entropy(probs).mean(0)
        return {"prob": mean, "entropy": total, "mutual_info": (total - aleatoric).clamp_min(0)}


def build_model(cfg: Dict) -> BAUNet:
    m = dict(cfg.get("model", {}))
    m.pop("name", None)
    return BAUNet(**m)
