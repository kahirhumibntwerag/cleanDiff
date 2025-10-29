from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.application.usecases.with_encoder_module import TrainEncoderDiffusionModule
from src.infrastructure.datamodules import create_clip_video_datamodule
from src.infrastructure.datasets.clip_video_dataset import ClipVideoDataset, ClipVideoDatasetConfig
from src.infrastructure.encoders import create_speed_encoder

from tests.fakes import FakeOptimizerBuilder, FakeSampler, FakeScheduler, FakeUNet, FakeVAE


def _make_temp_clip_dataset(tmp_path: Path, *, num_frames: int = 2, resolution: int = 32) -> Path:
    # Write speeds.csv
    speeds_csv = tmp_path / "speeds.csv"
    with speeds_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["clip_id", "speed"])
        writer.writerow(["clip_00000", 120.0])
    # Write clip folder and frames
    clip_dir = tmp_path / "clip_00000"
    clip_dir.mkdir(parents=True, exist_ok=True)
    for k in range(num_frames):
        arr = (np.random.rand(resolution, resolution, 3) * 255).astype(np.uint8)
        Image.fromarray(arr).save(clip_dir / f"{k:03d}.png")
    return tmp_path


def test_integration_dataset_to_training_step(tmp_path: Path) -> None:
    root = _make_temp_clip_dataset(tmp_path)
    cfg = ClipVideoDatasetConfig(root_dir=str(root), resolution=32, num_frames=2, center_crop=True)
    ds = ClipVideoDataset(cfg)
    loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=ClipVideoDataset.collate_fn)
    batch = next(iter(loader))
    # Collapse video to first frame to match current VAE image interface [B,3,H,W]
    batch["pixel_values"] = batch["pixel_values"][:, :, 0]

    # Models: use lightweight fakes for UNet/VAE/Scheduler/Sampler; real SpeedEncoder for conditioning
    vae = FakeVAE(latent_channels=4, scaling_factor=0.5)
    unet = FakeUNet(in_channels=4, out_channels=4)
    scheduler = FakeScheduler()
    opt_builder = FakeOptimizerBuilder()
    sampler = FakeSampler()

    encoder = create_speed_encoder(hidden_size=16)

    module = TrainEncoderDiffusionModule(
        vae=vae,
        unet=unet,
        scheduler=scheduler,
        encoder=encoder,
        optimizer_builder=opt_builder,
        sampler=sampler,
    )

    loss = module.training_step(batch, 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0


def test_integration_datamodule_factory_loaders(tmp_path: Path) -> None:
    root = _make_temp_clip_dataset(tmp_path)
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
    train_batch = next(iter(dm.train_dataloader()))
    val_batch = next(iter(dm.val_dataloader()))
    assert isinstance(train_batch["pixel_values"], torch.Tensor)
    assert isinstance(val_batch["pixel_values"], torch.Tensor)
    assert "encoder_inputs" in train_batch and "speed" in train_batch["encoder_inputs"]


