"""BAU-Net v2: Boundary- and Uncertainty-aware Network with point refinement.

Pipeline (see README for the figure)::

    image ─► encoder (residual | ResNet | ConvNeXt) ─► Global Context Transformer
          ─► attention-gated decoder (MC-dropout) ─► F (full resolution)
          ─► heads: boundary B̂, signed distance Ŝ, coarse mask M̂ (boundary-fused)
          ─► Uncertainty-guided Point Refinement ─► refined mask

All components can be switched off from the config, which is how the ablations are run;
with every option disabled the model reduces to a plain U-Net.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import AttentionGate, ConvBlock, ResidualBlock
from .encoders import build_encoder
from .point_refine import UncertaintyPointRefiner
from .transformer import GlobalContextTransformer


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
    """Boundary- and Uncertainty-aware segmentation network.

    Args:
        in_channels: input image channels.
        num_classes: foreground classes (1 for binary segmentation).
        encoder: ``"residual"`` (from scratch, uses ``widths``) or a torchvision backbone
            (``"resnet18"``, ``"resnet34"``, ``"resnet50"``, ``"convnext_tiny"``).
        pretrained: load ImageNet weights for torchvision encoders.
        widths: channel widths of the residual encoder (also the full-res stem width).
        decoder_max_width: cap on decoder channels (keeps large backbones affordable).
        bottleneck: ``"conv"`` or ``"transformer"`` (Global Context Transformer).
        transformer_dim / transformer_depth / transformer_heads: GCT size.
        attention: attention gates on the skip connections.
        boundary_head: auxiliary boundary branch fused into the mask branch.
        sdf_head: signed-distance-field regression head (shape prior).
        point_refine: Uncertainty-guided Point Refinement.
        residual: residual blocks in the decoder (and residual encoder).
        dropout: decoder spatial dropout (enables MC-dropout uncertainty).
        deep_supervision: auxiliary mask logits from intermediate decoder stages.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        encoder: str = "residual",
        pretrained: bool = False,
        widths: Sequence[int] = (32, 64, 128, 256, 512),
        decoder_max_width: int = 256,
        bottleneck: str = "conv",
        transformer_dim: int = 256,
        transformer_depth: int = 4,
        transformer_heads: int = 8,
        attention: bool = True,
        boundary_head: bool = True,
        sdf_head: bool = False,
        point_refine: bool = False,
        refine_train_points: int = 2048,
        refine_eval_points: int = 8192,
        refine_boundary_weight: float = 1.0,
        residual: bool = True,
        dropout: float = 0.1,
        deep_supervision: bool = False,
    ) -> None:
        super().__init__()
        if bottleneck not in ("conv", "transformer"):
            raise ValueError("bottleneck must be 'conv' or 'transformer'")
        self.num_classes = num_classes
        self.use_boundary = boundary_head
        self.use_sdf = sdf_head
        self.deep_supervision = deep_supervision

        self.encoder = build_encoder(encoder, in_channels, widths, residual, pretrained)
        enc_ch = self.encoder.out_channels
        self.context = (
            GlobalContextTransformer(enc_ch[-1], transformer_dim, transformer_depth, transformer_heads, dropout=dropout)
            if bottleneck == "transformer"
            else nn.Identity()
        )
        skip_ch = enc_ch[:-1][::-1]
        dec_ch = [min(c, decoder_max_width) for c in skip_ch]
        in_chs = [enc_ch[-1]] + dec_ch[:-1]
        self.decoder = nn.ModuleList(
            [DecoderStage(in_chs[i], skip_ch[i], dec_ch[i], attention, dropout, residual) for i in range(len(skip_ch))]
        )
        head_ch = dec_ch[-1]
        if boundary_head:
            self.boundary_head = nn.Sequential(
                nn.Conv2d(head_ch, head_ch // 2, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(head_ch // 2, 1, 1)
            )
            self.fuse = nn.Conv2d(head_ch + 1, head_ch, 3, padding=1)
        if sdf_head:
            self.sdf_head = nn.Sequential(
                nn.Conv2d(head_ch, head_ch // 2, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(head_ch // 2, 1, 1)
            )
        self.mask_head = nn.Conv2d(head_ch, num_classes, 1)
        if deep_supervision:
            self.aux_heads = nn.ModuleList([nn.Conv2d(c, num_classes, 1) for c in dec_ch[:-1]])
        self.refiner = (
            UncertaintyPointRefiner(
                head_ch, refine_train_points, refine_eval_points, boundary_weight=refine_boundary_weight
            )
            if point_refine
            else None
        )
        if point_refine and num_classes != 1:
            raise ValueError("point refinement is implemented for binary segmentation")
        self._init_weights()

    def _init_weights(self) -> None:
        for name, m in self.named_modules():
            if name.startswith("encoder.stages") or "refiner" in name or "context" in name:
                continue  # keep pretrained / purpose-specific initialisation
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats = self.encoder(x)
        y = self.context(feats[-1])
        skips = feats[:-1][::-1]
        aux: List[torch.Tensor] = []
        for i, (stage, skip) in enumerate(zip(self.decoder, skips, strict=True)):
            y = stage(y, skip)
            if self.deep_supervision and i < len(self.decoder) - 1:
                aux.append(self.aux_heads[i](y))
        out: Dict[str, torch.Tensor] = {}
        boundary = sdf = None
        if self.use_boundary:
            boundary = self.boundary_head(y)
            y = F.relu(self.fuse(torch.cat([y, torch.sigmoid(boundary)], dim=1))) + y
            out["boundary"] = boundary
        if self.use_sdf:
            sdf = torch.tanh(self.sdf_head(y))
            out["sdf"] = sdf
        mask = self.mask_head(y)
        if self.refiner is not None:
            out.update(self.refiner(y, mask, boundary, sdf))
        else:
            out["mask"] = mask
        if aux:
            out["aux"] = [F.interpolate(a, size=x.shape[-2:], mode="bilinear", align_corners=False) for a in aux]
        return out

    @torch.no_grad()
    def predict_with_uncertainty(
        self, x: torch.Tensor, n_samples: int = 10, tta: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Monte-Carlo dropout (optionally combined with flip test-time augmentation).

        Returns the mean foreground probability, predictive entropy (total uncertainty),
        mutual information (epistemic part) and the per-pixel standard deviation.
        """
        was_training = self.training
        self.eval()
        for m in self.modules():
            if isinstance(m, nn.Dropout2d):
                m.train()
        flips = [(), (-1,), (-2,), (-2, -1)] if tta else [()]
        probs = []
        for i in range(max(n_samples, 1)):
            dims = flips[i % len(flips)]
            xi = x.flip(dims) if dims else x
            p = torch.sigmoid(self(xi)["mask"])
            probs.append(p.flip(dims) if dims else p)
        self.train(was_training)
        probs_t = torch.stack(probs)
        mean = probs_t.mean(0)
        eps = 1e-8

        def entropy(p: torch.Tensor) -> torch.Tensor:
            return -(p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps))

        total = entropy(mean)
        aleatoric = entropy(probs_t).mean(0)
        return {
            "prob": mean,
            "entropy": total,
            "mutual_info": (total - aleatoric).clamp_min(0),
            "std": probs_t.std(0) if len(probs) > 1 else torch.zeros_like(mean),
        }


def build_model(cfg: Dict) -> BAUNet:
    m = dict(cfg.get("model", {}))
    m.pop("name", None)
    return BAUNet(**m)
