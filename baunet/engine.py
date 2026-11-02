"""Training and evaluation loops."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from .losses import BoundaryAwareLoss
from .metrics import SegmentationMeter
from .utils import AverageMeter, get_logger


def build_optimizer(model: torch.nn.Module, cfg: Dict) -> torch.optim.Optimizer:
    t = cfg["train"]
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or name.endswith(".bias") else decay).append(p)
    groups = [{"params": decay, "weight_decay": t.get("weight_decay", 1e-4)}, {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=t.get("lr", 1e-3), betas=(0.9, 0.999))


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: Dict, steps_per_epoch: int):
    t = cfg["train"]
    total = t["epochs"] * steps_per_epoch
    warmup = t.get("warmup_epochs", 0) * steps_per_epoch
    min_ratio = t.get("min_lr_ratio", 0.01)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        progress = (step - warmup) / max(total - warmup, 1)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    compute_hd95: bool = True,
    threshold: float = 0.5,
    mc_samples: int = 0,
    tta: bool = False,
) -> Dict[str, float]:
    """Evaluate a model; with ``mc_samples > 0`` predictions are MC-dropout (+TTA) averages and
    the uncertainty metrics (UEO, uncertainty AUROC) are reported as well."""
    model.eval()
    meter = SegmentationMeter(threshold=threshold, compute_hd95=compute_hd95)
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        if mc_samples > 0:
            res = model.predict_with_uncertainty(image, n_samples=mc_samples, tta=tta)
            meter.update_prob(res["prob"], mask, uncertainty=res["entropy"])
        else:
            meter.update(model(image)["mask"], mask)
    return meter.compute()


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: BoundaryAwareLoss,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: Dict,
        work_dir: str,
        device: Optional[torch.device] = None,
    ) -> None:
        self.cfg = cfg
        self.t = cfg["train"]
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = build_optimizer(model, cfg)
        self.scheduler = build_scheduler(self.optimizer, cfg, len(train_loader))
        self.use_amp = self.t.get("amp", True) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(log_file=str(self.work_dir / "train.log"))
        self.best_score = -float("inf")
        self.start_epoch = 1
        self.monitor = self.t.get("monitor", "dice")
        self.patience = self.t.get("early_stop_patience", 0)
        self._bad_epochs = 0
        (self.work_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    # ---------------------------------------------------------------- checkpoints
    def save(self, name: str, epoch: int) -> None:
        torch.save(
            {
                "epoch": epoch,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
                "best_score": self.best_score,
                "config": self.cfg,
            },
            self.work_dir / name,
        )

    def resume(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.scaler.load_state_dict(ckpt["scaler"])
        self.best_score = ckpt["best_score"]
        self.start_epoch = ckpt["epoch"] + 1
        self.logger.info(f"resumed from {path} (epoch {ckpt['epoch']})")

    # -------------------------------------------------------------------- loops
    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        meters: Dict[str, AverageMeter] = {}
        clip = self.t.get("grad_clip", 0.0)
        log_every = self.t.get("log_interval", 50)
        for step, batch in enumerate(self.train_loader, 1):
            image = batch["image"].to(self.device, non_blocking=True)
            mask = batch["mask"].to(self.device, non_blocking=True)
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(image)
            losses = self.criterion({k: _to_float(v) for k, v in outputs.items()}, mask)
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(losses["total"]).backward()
            if clip:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            for k, v in losses.items():
                meters.setdefault(k, AverageMeter()).update(v.item(), image.size(0))
            if step % log_every == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                self.logger.info(
                    f"epoch {epoch} [{step}/{len(self.train_loader)}] loss {meters['total'].avg:.4f} lr {lr:.2e}"
                )
        return {f"loss_{k}": m.avg for k, m in meters.items()}

    def fit(self) -> float:
        epochs = self.t["epochs"]
        eval_every = self.t.get("eval_interval", 1)
        history = self.work_dir / "history.jsonl"
        for epoch in range(self.start_epoch, epochs + 1):
            tic = time.time()
            row: Dict[str, float] = {"epoch": epoch, **self.train_one_epoch(epoch)}
            if epoch % eval_every == 0 or epoch == epochs:
                metrics = evaluate(
                    self.model, self.val_loader, self.device, compute_hd95=self.t.get("eval_hd95", False)
                )
                row.update({f"val_{k}": v for k, v in metrics.items()})
                score = metrics[self.monitor]
                if score > self.best_score:
                    self.best_score = score
                    self._bad_epochs = 0
                    self.save("best.pth", epoch)
                else:
                    self._bad_epochs += 1
            row["time_s"] = round(time.time() - tic, 1)
            with history.open("a") as f:
                f.write(json.dumps(row) + "\n")
            self.logger.info(" | ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()))
            self.save("last.pth", epoch)
            if self.patience and self._bad_epochs >= self.patience:
                self.logger.info(f"early stopping: no improvement in {self.patience} evaluations")
                break
        self.logger.info(f"best val {self.monitor}: {self.best_score:.4f}")
        return self.best_score


def _to_float(v):
    if isinstance(v, list):
        return [x.float() for x in v]
    return v.float() if v.is_floating_point() else v
