"""ONNX export of BAU-Net for deployment (ONNX Runtime, TensorRT, OpenVINO, ...).

The exported graph maps a normalised image batch ``(B, 3, H, W)`` to foreground
probabilities ``(B, 1, H, W)``. The spatial size is fixed at export time: the Global
Context Transformer's sine-cosine positional encoding and the point-refinement budget
depend on the feature-map size. The batch dimension can be dynamic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from .models import BAUNet


class ProbabilityHead(nn.Module):
    """Wraps BAU-Net so the exported graph returns only ``sigmoid(mask)``."""

    def __init__(self, model: BAUNet, refine: bool = True) -> None:
        super().__init__()
        self.model = model
        self.refine = refine

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        refiner = self.model.refiner
        if not self.refine:
            self.model.refiner = None
        try:
            out = self.model(x)
        finally:
            self.model.refiner = refiner
        return torch.sigmoid(out["mask"])


def export_onnx(
    model: BAUNet,
    path: str,
    size: Sequence[int] = (352, 352),
    opset: int = 17,
    dynamic_batch: bool = True,
    refine: bool = True,
) -> Path:
    """Export ``model`` (put in eval mode) to ``path`` and return the path."""
    model = model.eval()
    wrapper = ProbabilityHead(model, refine=refine).eval()
    in_ch = model.encoder.stem.body[0].in_channels if hasattr(model.encoder, "stem") else 3
    dummy = torch.randn(1, in_ch, int(size[0]), int(size[1]))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    axes = {"image": {0: "batch"}, "probability": {0: "batch"}} if dynamic_batch else None
    # nn.TransformerEncoderLayer uses a fused "fast path" kernel in eval mode that has no ONNX
    # equivalent; disable it while tracing so the standard attention ops are exported.
    fastpath = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        _export(wrapper, dummy, out, axes, opset)
    finally:
        torch.backends.mha.set_fastpath_enabled(fastpath)
    return out


def _export(wrapper: nn.Module, dummy: torch.Tensor, out: Path, axes, opset: int) -> None:
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy,),
            str(out),
            input_names=["image"],
            output_names=["probability"],
            dynamic_axes=axes,
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )


def calibrate_batchnorm(model: BAUNet, size: Sequence[int], steps: int = 10, batch: int = 2) -> BAUNet:
    """Populate BatchNorm running statistics of an *untrained* model with random inputs.

    Freshly initialised networks have running mean 0 / variance 1, so activations grow
    layer after layer in eval mode and the exported graph cannot be compared meaningfully
    with PyTorch. Only used when exporting a model built from a config without weights.
    """
    in_ch = model.encoder.stem.body[0].in_channels if hasattr(model.encoder, "stem") else 3
    model.train()
    with torch.no_grad():
        for _ in range(steps):
            model(torch.randn(batch, in_ch, int(size[0]), int(size[1])))
    return model.eval()


def verify_onnx(path: str, model: BAUNet, x: torch.Tensor, refine: bool = True) -> dict:
    """Run the exported graph with ONNX Runtime and compare it with PyTorch.

    Returns ``{"max_abs_diff", "mismatch_fraction"}`` where the latter is the fraction of
    pixels differing by more than 1e-3. With point refinement a handful of pixels exactly at
    the top-k cut-off may legitimately be selected differently under float32 rounding.
    """
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    (onnx_out,) = sess.run(None, {"image": x.numpy().astype(np.float32)})
    with torch.no_grad():
        ref = ProbabilityHead(model.eval(), refine=refine)(x).numpy()
    diff = np.abs(onnx_out - ref)
    return {"max_abs_diff": float(diff.max()), "mismatch_fraction": float((diff > 1e-3).mean())}
