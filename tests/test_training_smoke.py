from baunet.data import build_dataloaders
from baunet.engine import Trainer
from baunet.losses import build_loss
from baunet.models import build_model
from baunet.utils import load_config, merge_overrides, seed_everything


def test_one_epoch_on_synthetic(tmp_path):
    seed_everything(0)
    cfg = merge_overrides(load_config("configs/synthetic.yaml"), ["train.epochs=1", "data.train_size=32"])
    train_loader, val_loader = build_dataloaders(cfg)
    trainer = Trainer(build_model(cfg), build_loss(cfg), train_loader, val_loader, cfg, str(tmp_path))
    score = trainer.fit()
    assert 0.0 <= score <= 1.0
    assert (tmp_path / "best.pth").exists() and (tmp_path / "history.jsonl").exists()
