import numpy as np
import pytest
import torch

from baunet.losses import BoundaryAwareLoss, DiceLoss, mask_to_boundary
from baunet.metrics import SegmentationMeter, dice_score, hd95, iou_score


def square_mask(size=32, lo=8, hi=24):
    m = torch.zeros(1, 1, size, size)
    m[..., lo:hi, lo:hi] = 1
    return m


def test_boundary_is_thin_ring():
    b = mask_to_boundary(square_mask(), width=3)
    assert b[0, 0, 16, 16] == 0  # interior
    assert b[0, 0, 0, 0] == 0  # far background
    assert b[0, 0, 8, 16] == 1  # on the edge
    assert 0 < b.mean() < 0.3


def test_dice_loss_near_zero_for_perfect_prediction():
    m = square_mask()
    logits = (m * 2 - 1) * 20
    assert DiceLoss()(logits, m).item() < 1e-3


def test_boundary_aware_loss_terms():
    m = square_mask()
    outputs = {"mask": torch.randn_like(m), "boundary": torch.randn_like(m), "aux": [torch.randn_like(m)]}
    losses = BoundaryAwareLoss()(outputs, m)
    assert set(losses) == {"seg", "boundary", "aux", "total"}
    assert torch.isclose(losses["total"], losses["seg"] + losses["boundary"] + losses["aux"])


def test_loss_backpropagates():
    m = square_mask()
    logits = torch.zeros_like(m, requires_grad=True)
    BoundaryAwareLoss()({"mask": logits, "boundary": logits}, m)["total"].backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0


def test_dice_iou_values():
    pred = square_mask(lo=8, hi=24)
    target = square_mask(lo=8, hi=16)
    # |P| = 256, |T| = 64, |P∩T| = 64
    assert dice_score(pred, target).item() == pytest.approx(2 * 64 / (256 + 64))
    assert iou_score(pred, target).item() == pytest.approx(64 / 256)


def test_empty_masks_count_as_perfect():
    z = torch.zeros(1, 1, 8, 8)
    assert dice_score(z, z).item() == 1.0
    assert hd95(z[0, 0].numpy(), z[0, 0].numpy()) == 0.0


def test_hd95_of_shifted_square():
    a = np.zeros((64, 64), bool)
    b = np.zeros((64, 64), bool)
    a[10:30, 10:30] = True
    b[13:33, 10:30] = True
    assert hd95(a, b) == pytest.approx(3.0)


def test_meter_aggregates():
    meter = SegmentationMeter()
    m = square_mask()
    meter.update((m * 2 - 1) * 10, m)
    res = meter.compute()
    assert res["dice"] == pytest.approx(1.0)
    assert res["hd95"] == pytest.approx(0.0)
