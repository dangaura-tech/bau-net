import numpy as np
import pytest
import torch

from baunet.geometry import signed_distance, signed_distance_np
from baunet.losses import BoundaryAwareLoss, DiceLoss, mask_to_boundary
from baunet.metrics import (
    SegmentationMeter,
    auroc,
    boundary_iou,
    dice_score,
    expected_calibration_error,
    hd95,
    iou_score,
    uncertainty_error_overlap,
)


def square_mask(size=32, lo=8, hi=24):
    m = torch.zeros(1, 1, size, size)
    m[..., lo:hi, lo:hi] = 1
    return m


def test_boundary_is_thin_ring():
    b = mask_to_boundary(square_mask(), width=3)
    assert b[0, 0, 16, 16] == 0 and b[0, 0, 0, 0] == 0 and b[0, 0, 8, 16] == 1
    assert 0 < b.mean() < 0.3


def test_signed_distance_sign_and_range():
    sdf = signed_distance_np(square_mask()[0, 0].numpy())
    assert sdf[16, 16] == pytest.approx(-1.0)  # deepest interior point
    assert sdf[0, 0] == pytest.approx(1.0)  # farthest exterior point
    assert sdf.min() >= -1 and sdf.max() <= 1
    assert signed_distance_np(np.zeros((4, 4))).min() == 1.0
    assert signed_distance(square_mask()).shape == (1, 1, 32, 32)


def test_dice_loss_near_zero_for_perfect_prediction():
    m = square_mask()
    assert DiceLoss()((m * 2 - 1) * 20, m).item() < 1e-3


def test_full_loss_terms_and_backprop():
    m = square_mask()
    logits = torch.zeros_like(m, requires_grad=True)
    outputs = {
        "mask": logits,
        "boundary": torch.randn_like(m),
        "sdf": torch.tanh(torch.randn_like(m)),
        "aux": [torch.randn_like(m)],
        "point_coords": torch.rand(1, 20, 2),
        "point_logits": logits.flatten(2)[:, :, :20],
    }
    losses = BoundaryAwareLoss()(outputs, m)
    assert set(losses) == {"seg", "boundary", "sdf", "consistency", "point", "aux", "total"}
    parts = sum(v for k, v in losses.items() if k != "total")
    assert torch.isclose(losses["total"], parts)
    losses["total"].backward()
    assert logits.grad.abs().sum() > 0


def test_consistency_is_small_for_matching_mask_and_sdf():
    m = square_mask()
    sdf = signed_distance(m)
    loss = BoundaryAwareLoss(sdf_weight=0, consistency_weight=1.0, point_weight=0, boundary_weight=0, aux_weight=0)
    matching = loss({"mask": (m * 2 - 1) * 20, "sdf": sdf}, m)["consistency"]
    wrong = loss({"mask": (m * 2 - 1) * -20, "sdf": sdf}, m)["consistency"]
    assert matching < 0.1 < wrong


def test_dice_iou_values():
    pred, target = square_mask(lo=8, hi=24), square_mask(lo=8, hi=16)
    assert dice_score(pred, target).item() == pytest.approx(2 * 64 / (256 + 64))
    assert iou_score(pred, target).item() == pytest.approx(64 / 256)


def test_boundary_iou_penalises_contour_errors_more_than_iou():
    target = square_mask(64, 8, 56)
    pred = target.clone()
    pred[..., 8:10, 8:56] = 0  # shave two rows off the contour
    assert boundary_iou(pred, target).item() < iou_score(pred, target).item()
    assert boundary_iou(target, target).item() == pytest.approx(1.0)


def test_hd95_and_empty_masks():
    a = np.zeros((64, 64), bool)
    b = np.zeros((64, 64), bool)
    a[10:30, 10:30] = True
    b[13:33, 10:30] = True
    assert hd95(a, b) == pytest.approx(3.0)
    z = torch.zeros(1, 1, 8, 8)
    assert dice_score(z, z).item() == 1.0 and hd95(z[0, 0].numpy(), z[0, 0].numpy()) == 0.0


def test_uncertainty_metrics():
    err = np.zeros((16, 16), bool)
    err[4:8, 4:8] = True
    perfect = err.astype(float)
    assert uncertainty_error_overlap(perfect, err) == pytest.approx(1.0)
    assert auroc(perfect, err) == pytest.approx(1.0)
    assert auroc(np.ones((16, 16)), err) == pytest.approx(0.5)
    assert expected_calibration_error(np.full(100, 0.7), np.r_[np.ones(70), np.zeros(30)]) == pytest.approx(0.0)


def test_meter_aggregates_with_uncertainty():
    meter = SegmentationMeter()
    m = square_mask()
    meter.update((m * 2 - 1) * 10, m, uncertainty=torch.rand_like(m))
    res = meter.compute()
    assert res["dice"] == pytest.approx(1.0) and res["boundary_iou"] == pytest.approx(1.0)
    assert res["hd95"] == pytest.approx(0.0)
    assert np.isnan(res["ueo"])  # no errors -> undefined
