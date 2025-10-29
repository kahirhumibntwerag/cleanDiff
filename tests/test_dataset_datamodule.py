from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import torch

from PIL import Image

from src.infrastructure.datasets.clip_video_dataset import ClipVideoDataset, ClipVideoDatasetConfig
from src.infrastructure.datamodules import create_clip_video_datamodule


def _write_clip(root: Path, clip_id: str, *, num_frames: int = 3, size: tuple[int, int] = (32, 24)) -> None:
    clip_dir = root / clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    W, H = size
    for k in range(num_frames):
        arr = (np.random.rand(H, W, 3) * 255).astype(np.uint8)
        Image.fromarray(arr).save(clip_dir / f"{k:03d}.png")


def _write_speeds_csv(path: Path, rows: list[tuple[str, float]]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip_id", "speed"])
        for cid, sp in rows:
            w.writerow([cid, sp])


def test_clip_video_dataset_happy_path_and_shapes(tmp_path: Path) -> None:
    root = tmp_path
    _write_clip(root, "clip_00000", num_frames=2, size=(40, 30))
    _write_speeds_csv(root / "speeds.csv", [("clip_00000", 123.0)])

    cfg = ClipVideoDatasetConfig(root_dir=str(root), resolution=32, num_frames=2, center_crop=True)
    ds = ClipVideoDataset(cfg)
    sample = ds[0]
    x = sample["pixel_values"]
    sp = sample["encoder_inputs"]["speed"]
    assert x.shape == (3, 2, 32, 32)
    assert sp.shape == (1,)
    assert torch.isfinite(x).all()
    assert x.min() >= -1.0 - 1e-5 and x.max() <= 1.0 + 1e-5


def test_clip_video_dataset_missing_speeds_raises(tmp_path: Path) -> None:
    root = tmp_path
    _write_clip(root, "clip_00000", num_frames=1)
    with pytest.raises(FileNotFoundError):
        ClipVideoDataset(ClipVideoDatasetConfig(root_dir=str(root)))


def test_clip_video_dataset_wrong_header_raises(tmp_path: Path) -> None:
    root = tmp_path
    _write_clip(root, "clip_00000", num_frames=1)
    with (root / "speeds.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "value"])  # wrong headers
        w.writerow(["clip_00000", 1.0])
    with pytest.raises(ValueError):
        ClipVideoDataset(ClipVideoDatasetConfig(root_dir=str(root)))


def test_clip_video_dataset_missing_frame_raises(tmp_path: Path) -> None:
    root = tmp_path
    # only write frame 0, but request 2 frames
    _write_clip(root, "clip_00000", num_frames=1)
    _write_speeds_csv(root / "speeds.csv", [("clip_00000", 10.0)])
    ds = ClipVideoDataset(ClipVideoDatasetConfig(root_dir=str(root), num_frames=2))
    with pytest.raises(FileNotFoundError):
        _ = ds[0]


def test_clip_video_datamodule_loaders_return_expected_shapes(tmp_path: Path) -> None:
    root = tmp_path
    _write_clip(root, "clip_00000", num_frames=2, size=(28, 32))
    _write_clip(root, "clip_00001", num_frames=2, size=(28, 32))
    _write_speeds_csv(root / "speeds.csv", [("clip_00000", 42.0), ("clip_00001", 43.0)])

    dm = create_clip_video_datamodule(
        train_root=str(root),
        val_root=str(root),
        batch_size=2,
        num_workers=0,
        resolution=32,
        num_frames=2,
        center_crop=True,
    )
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    assert batch["pixel_values"].shape == (2, 3, 2, 32, 32)
    assert batch["encoder_inputs"]["speed"].shape == (2, 1)


