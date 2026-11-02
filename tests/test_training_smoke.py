import math

from baunet.data import build_dataloaders
from baunet.engine import Trainer, evaluate
from baunet.losses import build_loss
from baunet.models import build_model
from baunet.utils import load_config, merge_overrides, seed_everything


def test_one_epoch_on_synthetic_with_all_components(tmp_path):
    seed_everything(0)
    cfg = merge_overrides(load_config("configs/synthetic.yaml"), ["train.epochs=1", "data.train_size=32"])
    assert cfg["model"]["point_refine"] and cfg["model"]["sdf_head"] and cfg["model"]["bottleneck"] == "transformer"
    train_loader, val_loader = build_dataloaders(cfg)
    model = build_model(cfg)
    trainer = Trainer(model, build_loss(cfg), train_loader, val_loader, cfg, str(tmp_path))
    score = trainer.fit()
    assert 0.0 <= score <= 1.0
    assert (tmp_path / "best.pth").exists() and (tmp_path / "history.jsonl").exists()
    res = evaluate(model, val_loader, trainer.device, compute_hd95=False, mc_samples=2, tta=True)
    assert {"dice", "boundary_iou", "ece", "ueo", "unc_auroc"} <= set(res)
    assert all(math.isnan(v) or 0 <= v <= 1 for k, v in res.items())
