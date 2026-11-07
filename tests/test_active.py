import numpy as np
import pytest
import torch

from baunet.active import (
    acquisition_scores,
    bottleneck_embedding,
    diverse_select,
    k_center_greedy,
    topk_select,
)
from baunet.models import build_model


def _res(prob, mi):
    eps = 1e-8
    ent = -(prob * torch.log(prob + eps) + (1 - prob) * torch.log(1 - prob + eps))
    return {"prob": prob, "entropy": ent, "mutual_info": mi}


def test_boundary_score_focuses_on_contours():
    prob = torch.zeros(2, 1, 32, 32)
    prob[:, :, 8:24, 8:24] = 0.95
    mi = torch.zeros(2, 1, 32, 32)
    mi[0, :, 7:9, 8:24] = 0.6  # image 0: uncertainty on the object edge
    mi[1, :, 0:2, 0:16] = 0.6  # image 1: the same amount far away in the background
    s = acquisition_scores(_res(prob, mi), band_width=5)
    assert s["bald"][0] == pytest.approx(s["bald"][1].item())  # global mean cannot tell them apart
    assert s["boundary"][0] > 3 * s["boundary"][1]
    assert set(s) == {"entropy", "bald", "boundary", "region", "random"}


def test_region_score_reacts_to_a_small_bad_area_and_empty_masks_fall_back():
    prob = torch.full((2, 1, 40, 40), 0.02)
    mi = torch.full((2, 1, 40, 40), 0.01)
    mi[0, :, :4, :4] = 0.8  # 1 % of pixels very uncertain
    s = acquisition_scores(_res(prob, mi), region_quantile=0.01)
    assert s["region"][0] > 10 * s["region"][1]
    torch.testing.assert_close(s["boundary"], s["bald"])  # no predicted foreground -> global BALD


def test_k_center_greedy_covers_clusters():
    rng = np.random.default_rng(0)
    centres = np.array([[0, 0], [10, 0], [0, 10], [10, 10]], float)
    pts = np.concatenate([c + rng.normal(0, 0.3, (25, 2)) for c in centres])
    picked = k_center_greedy(pts, 4)
    assert len(set(picked.tolist())) == 4
    assert sorted({int(i) // 25 for i in picked}) == [0, 1, 2, 3]  # one per cluster
    # seeding with labelled points from clusters 0-2 makes the first pick come from cluster 3
    assert k_center_greedy(pts, 1, labelled=centres[:3])[0] // 25 == 3


def test_diverse_select_avoids_near_duplicates():
    emb = np.array([[0, 0], [0.01, 0], [0.02, 0], [5, 5], [9, 0]], float)
    scores = np.array([0.9, 0.89, 0.88, 0.5, 0.4])
    assert topk_select(scores, 2).tolist() == [0, 1]  # duplicates
    pick = diverse_select(scores, emb, 2, pool_factor=2.5)
    assert len(pick) == 2 and not ({0, 1} <= set(pick.tolist()) or {1, 2} <= set(pick.tolist()))
    assert set(pick.tolist()) <= {0, 1, 2, 3, 4}


def test_scores_and_embeddings_from_a_real_model():
    torch.manual_seed(0)
    cfg = {"model": {"encoder": "residual", "widths": [8, 16, 32, 64, 64], "transformer_dim": 32,
                     "transformer_depth": 1, "transformer_heads": 2, "dropout": 0.2}}  # fmt: skip
    model = build_model(cfg)
    x = torch.randn(3, 3, 64, 64)
    res = model.predict_with_uncertainty(x, n_samples=4)
    s = acquisition_scores(res)
    for v in s.values():
        assert v.shape == (3,) and torch.isfinite(v).all()
    assert (s["bald"] >= 0).all()
    e = bottleneck_embedding(model, x)
    assert e.shape[0] == 3
    torch.testing.assert_close(e.norm(dim=1), torch.ones(3))
