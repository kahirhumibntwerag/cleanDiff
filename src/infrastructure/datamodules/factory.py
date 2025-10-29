from __future__ import annotations

from typing import Optional

from .clip_video_datamodule import ClipVideoDataModule


def create_clip_video_datamodule(
    *,
    train_root: str,
    val_root: Optional[str] = None,
    batch_size: int = 2,
    num_workers: int = 4,
    resolution: int = 256,
    num_frames: int = 8,
    center_crop: bool = True,
) -> ClipVideoDataModule:
    return ClipVideoDataModule(
        train_root=train_root,
        val_root=val_root,
        batch_size=batch_size,
        num_workers=num_workers,
        resolution=resolution,
        num_frames=num_frames,
        center_crop=center_crop,
    )


