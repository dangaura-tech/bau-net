import pytest
import torch

from baunet.models import BAUNet, GlobalContextTransformer, UncertaintyPointRefiner, point_sample
from baunet.models.point_refine import sample_training_points, top_uncertain_pixels
from baunet.models.transformer import sincos_2d

SMALL = dict(widths=(8, 16, 32), transformer_dim=32, transformer_depth=1, transformer_heads=4)


@pytest.mark.parametrize("deep_supervision", [False, True])
@pytest.mark.parametrize("boundary", [False, True])
def test_output_shapes(boundary, deep_supervision):
    model = BAUNet(boundary_head=boundary, deep_supervision=deep_supervision, **SMALL)
    out = model(torch.randn(2, 3, 64, 48))
    assert out["mask"].shape == (2, 1, 64, 48)
    assert ("boundary" in out) == boundary
    if deep_supervision:
        assert all(a.shape == (2, 1, 64, 48) for a in out["aux"])


def test_full_v2_model_train_and_eval_paths():
    model = BAUNet(bottleneck="transformer", sdf_head=True, point_refine=True,
                   refine_train_points=32, refine_eval_points=100, **SMALL)  # fmt: skip
    out = model.train()(torch.randn(2, 3, 32, 32))
    assert out["point_coords"].shape == (2, 32, 2) and out["point_logits"].shape == (2, 1, 32)
    assert out["sdf"].abs().max() <= 1
    out = model.eval()(torch.randn(1, 3, 37, 41))
    assert out["mask"].shape == out["mask_coarse"].shape == (1, 1, 37, 41)
    changed = (out["mask"] != out["mask_coarse"]).sum()
    assert changed <= 100  # only the selected pixels may change


@pytest.mark.parametrize("encoder", ["resnet18", "convnext_tiny"])
def test_torchvision_encoders(encoder):
    model = BAUNet(encoder=encoder, pretrained=False, widths=(16,), bottleneck="transformer",
                   transformer_dim=32, transformer_depth=1, transformer_heads=4)  # fmt: skip
    assert model.eval()(torch.randn(1, 3, 64, 64))["mask"].shape == (1, 1, 64, 64)


def test_plain_unet_variant():
    model = BAUNet(widths=(8, 16), attention=False, boundary_head=False, residual=False, dropout=0.0)
    assert set(model(torch.randn(1, 3, 32, 32))) == {"mask"}


def test_transformer_starts_as_identity():
    gct = GlobalContextTransformer(16, dim=32, depth=1, heads=4)
    x = torch.randn(2, 16, 5, 7)
    assert torch.allclose(gct(x), x)
    assert sincos_2d(5, 7, 32).shape == (35, 32)


def test_point_sampling_focuses_on_uncertain_region():
    torch.manual_seed(0)
    score = torch.zeros(1, 1, 64, 64)
    score[..., :, 24:40] = 1.0  # uncertain vertical band covering 25 % of the image
    coords = sample_training_points(score, 200, oversample=4, importance=0.75)
    xs = coords[0, :, 0] * 64
    in_band = (xs > 23) & (xs < 41)
    assert in_band[:150].float().mean() > 0.95  # importance samples come from the band
    assert in_band[150:].float().mean() < 0.6  # the uniform remainder covers the whole image
    idx, c = top_uncertain_pixels(score, 10)
    assert torch.all((c[..., 0] * 64 > 24) & (c[..., 0] * 64 < 40))


def test_point_sample_matches_pixel_centres():
    x = torch.arange(16.0).view(1, 1, 4, 4)
    coords = torch.tensor([[[0.125, 0.125], [0.875, 0.625]]])  # pixel (0,0) and (row 2, col 3)
    assert point_sample(x, coords)[0, 0].tolist() == [0.0, 11.0]


def test_refiner_identity_at_init():
    ref = UncertaintyPointRefiner(8, train_points=16, eval_points=50).eval()
    mask = torch.randn(1, 1, 16, 16)
    out = ref(torch.randn(1, 8, 16, 16), mask)
    assert torch.allclose(out["mask"], mask, atol=1e-6)


def test_mc_dropout_uncertainty_is_bounded():
    torch.manual_seed(0)
    model = BAUNet(dropout=0.3, **SMALL).eval()
    res = model.predict_with_uncertainty(torch.randn(1, 3, 32, 32), n_samples=4, tta=True)
    assert res["prob"].shape == (1, 1, 32, 32)
    assert torch.all((res["prob"] >= 0) & (res["prob"] <= 1))
    assert torch.all(res["mutual_info"] >= 0) and torch.all(res["std"] >= 0)
    assert torch.all(res["entropy"] <= torch.log(torch.tensor(2.0)) + 1e-5)
    assert not model.training  # original mode is restored
    model.train()
    model.predict_with_uncertainty(torch.randn(1, 3, 32, 32), n_samples=2)
    assert model.training


def test_mc_dropout_samples_differ():
    torch.manual_seed(0)
    model = BAUNet(dropout=0.5, **SMALL)
    res = model.predict_with_uncertainty(torch.randn(1, 3, 32, 32), n_samples=8)
    assert res["mutual_info"].sum() > 0
