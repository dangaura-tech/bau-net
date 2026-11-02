import pytest
import torch

from baunet.models import BAUNet


@pytest.mark.parametrize("deep_supervision", [False, True])
@pytest.mark.parametrize("boundary", [False, True])
def test_output_shapes(boundary, deep_supervision):
    model = BAUNet(widths=(8, 16, 32), boundary_head=boundary, deep_supervision=deep_supervision)
    out = model(torch.randn(2, 3, 64, 48))
    assert out["mask"].shape == (2, 1, 64, 48)
    assert ("boundary" in out) == boundary
    if boundary:
        assert out["boundary"].shape == (2, 1, 64, 48)
    if deep_supervision:
        assert all(a.shape == (2, 1, 64, 48) for a in out["aux"])


def test_handles_odd_input_sizes():
    model = BAUNet(widths=(8, 16, 32, 64))
    assert model(torch.randn(1, 3, 75, 61))["mask"].shape == (1, 1, 75, 61)


def test_plain_unet_variant():
    model = BAUNet(widths=(8, 16), attention=False, boundary_head=False, residual=False, dropout=0.0)
    assert set(model(torch.randn(1, 3, 32, 32))) == {"mask"}


def test_mc_dropout_uncertainty_is_bounded():
    torch.manual_seed(0)
    model = BAUNet(widths=(8, 16, 32), dropout=0.3).eval()
    res = model.predict_with_uncertainty(torch.randn(1, 3, 32, 32), n_samples=4)
    assert res["prob"].shape == (1, 1, 32, 32)
    assert torch.all((res["prob"] >= 0) & (res["prob"] <= 1))
    assert torch.all(res["mutual_info"] >= 0)
    assert torch.all(res["entropy"] <= torch.log(torch.tensor(2.0)) + 1e-5)
    assert not model.training  # original mode is restored
    model.train()
    model.predict_with_uncertainty(torch.randn(1, 3, 32, 32), n_samples=2)
    assert model.training


def test_mc_dropout_samples_differ():
    torch.manual_seed(0)
    model = BAUNet(widths=(8, 16, 32), dropout=0.5)
    res = model.predict_with_uncertainty(torch.randn(1, 3, 32, 32), n_samples=8)
    assert res["mutual_info"].sum() > 0  # stochastic passes disagree somewhere
