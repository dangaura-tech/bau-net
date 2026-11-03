import pytest
import torch

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from baunet.export import ProbabilityHead, calibrate_batchnorm, export_onnx, verify_onnx  # noqa: E402
from baunet.models import BAUNet  # noqa: E402


def tiny(refine: bool) -> BAUNet:
    torch.manual_seed(0)
    return BAUNet(widths=(8, 16, 32), bottleneck="transformer", transformer_dim=32, transformer_depth=1,
                  transformer_heads=4, sdf_head=True, point_refine=refine, refine_eval_points=64).eval()  # fmt: skip


@pytest.mark.parametrize("refine", [False, True])
def test_onnx_matches_pytorch(tmp_path, refine):
    model = tiny(refine)
    path = export_onnx(model, str(tmp_path / "m.onnx"), size=(64, 64), refine=refine)
    assert path.exists()
    stats = verify_onnx(str(path), model, torch.randn(3, 3, 64, 64), refine=refine)  # dynamic batch of 3
    assert stats["max_abs_diff"] < 1e-4 and stats["mismatch_fraction"] == 0.0


def test_probability_head_restores_refiner():
    model = tiny(True)
    head = ProbabilityHead(model, refine=False)
    out = head(torch.randn(1, 3, 32, 32))
    assert out.shape == (1, 1, 32, 32) and 0 <= out.min() and out.max() <= 1
    assert model.refiner is not None


def test_calibrate_batchnorm_updates_running_statistics():
    torch.manual_seed(0)
    model = BAUNet(widths=(8, 16, 32)).eval()
    bns = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    assert all(int(b.num_batches_tracked) == 0 for b in bns)
    calibrate_batchnorm(model, (32, 32), steps=3)
    assert all(int(b.num_batches_tracked) == 3 for b in bns)
    assert any(b.running_mean.abs().sum() > 0 for b in bns)
    assert not model.training
