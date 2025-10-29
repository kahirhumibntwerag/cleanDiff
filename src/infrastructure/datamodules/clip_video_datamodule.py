from __future__ import annotations

from typing import Optional

from torch.utils.data import DataLoader

from src.domain.interfaces.datamodule import DiffusionDataModule
from src.domain.interfaces.types import DiffusionBatch
from src.infrastructure.datasets.clip_video_dataset import (
    ClipVideoDataset,
    ClipVideoDatasetConfig,
)


class ClipVideoDataModule(DiffusionDataModule):
    def __init__(
        self,
        *,
        train_root: str,
        val_root: Optional[str] = None,
        batch_size: int = 2,
        num_workers: int = 4,
        resolution: int = 256,
        num_frames: int = 8,
        center_crop: bool = True,
    ) -> None:
        self._batch_size = int(batch_size)
        self._num_workers = int(num_workers)
        self._train_root = train_root
        self._val_root = val_root or train_root
        self._resolution = int(resolution)
        self._num_frames = int(num_frames)
        self._center_crop = bool(center_crop)

        self._train_ds: Optional[ClipVideoDataset] = None
        self._val_ds: Optional[ClipVideoDataset] = None

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def setup(self, stage: Optional[str] = None) -> None:
        train_cfg = ClipVideoDatasetConfig(
            root_dir=self._train_root,
            resolution=self._resolution,
            num_frames=self._num_frames,
            center_crop=self._center_crop,
        )
        val_cfg = ClipVideoDatasetConfig(
            root_dir=self._val_root,
            resolution=self._resolution,
            num_frames=self._num_frames,
            center_crop=self._center_crop,
        )
        self._train_ds = ClipVideoDataset(train_cfg)
        self._val_ds = ClipVideoDataset(val_cfg)

    def train_dataloader(self) -> DataLoader[DiffusionBatch]:
        assert self._train_ds is not None, "Call setup() before requesting dataloaders"
        return DataLoader(
            self._train_ds,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            shuffle=True,
            collate_fn=ClipVideoDataset.collate_fn,
        )

    def val_dataloader(self) -> DataLoader[DiffusionBatch]:
        assert self._val_ds is not None, "Call setup() before requesting dataloaders"
        return DataLoader(
            self._val_ds,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            shuffle=False,
            collate_fn=ClipVideoDataset.collate_fn,
        )

    def test_dataloader(self) -> DataLoader[DiffusionBatch]:
        return self.val_dataloader()

    def predict_dataloader(self) -> DataLoader[DiffusionBatch]:
        return self.val_dataloader()


