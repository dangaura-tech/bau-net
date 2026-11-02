import numpy as np
import torch
from PIL import Image

from baunet.data import FolderSegmentationDataset, SyntheticShapesDataset, build_transforms
from baunet.utils import load_config, merge_overrides


def test_synthetic_dataset_is_deterministic():
    a = SyntheticShapesDataset(n=4, size=32, seed=3)[2]
    b = SyntheticShapesDataset(n=4, size=32, seed=3)[2]
    assert torch.equal(a["image"], b["image"]) and torch.equal(a["mask"], b["mask"])
    assert set(a["mask"].unique().tolist()) <= {0.0, 1.0}


def test_train_transforms_keep_mask_binary():
    tf = build_transforms({"image_size": [48, 48]}, train=True)
    sample = SyntheticShapesDataset(n=1, size=64)[0]
    img, mask = tf(sample["image"], sample["mask"])
    assert img.shape == (3, 48, 48) and mask.shape == (1, 48, 48)
    assert set(mask.unique().tolist()) <= {0.0, 1.0}


def test_folder_dataset(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "masks").mkdir()
    for i in range(3):
        Image.fromarray(np.full((20, 30, 3), i * 40, np.uint8)).save(tmp_path / "images" / f"img{i}.jpg")
        m = np.zeros((20, 30), np.uint8)
        m[5:15, 5:15] = 255
        Image.fromarray(m).save(tmp_path / "masks" / f"img{i}.png")
    split = tmp_path / "train.txt"
    split.write_text("img0\nimg2\n")
    ds = FolderSegmentationDataset(str(tmp_path), str(split))
    assert len(ds) == 2
    item = ds[1]
    assert item["name"] == "img2"
    assert item["image"].shape == (3, 20, 30)
    assert item["mask"].sum() == 100


def test_config_inheritance_and_overrides():
    cfg = load_config("configs/ablations/kvasir_no_boundary.yaml")
    assert cfg["model"]["boundary_head"] is False
    assert cfg["model"]["attention"] is True  # inherited from _base_
    assert cfg["model"]["point_refine"] is True and cfg["model"]["encoder"] == "resnet34"
    assert cfg["data"]["root"] == "data/kvasir"
    cfg = merge_overrides(cfg, ["train.lr=5e-4", "model.widths=[16,32]"])
    assert cfg["train"]["lr"] == 5e-4 and cfg["model"]["widths"] == [16, 32]


def test_every_config_builds_a_model():
    import glob

    import torch

    from baunet.models import build_model

    for path in sorted(glob.glob("configs/**/*.yaml", recursive=True)):
        if path.endswith("_base_.yaml"):
            continue
        cfg = load_config(path)
        cfg["model"]["pretrained"] = False  # no downloads in tests
        model = build_model(cfg).eval()
        with torch.no_grad():
            assert model(torch.randn(1, 3, 64, 64))["mask"].shape == (1, 1, 64, 64), path
