from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from torch.utils.data import DataLoader

from .types import DiffusionBatch


@runtime_checkable
class DiffusionDataModule(Protocol):
    """
    DataModule protocol for diffusion training. Mirrors the minimal Lightning
    DataModule contract but types DataLoader items as `DiffusionBatch`.
    """

    @property
    def batch_size(self) -> int:
        ...

    def setup(self, stage: Optional[str] = None) -> None:
        ...

    def train_dataloader(self) -> DataLoader[DiffusionBatch]:
        ...

    def val_dataloader(self) -> DataLoader[DiffusionBatch]:
        ...

    def test_dataloader(self) -> DataLoader[DiffusionBatch]:
        ...

    def predict_dataloader(self) -> DataLoader[DiffusionBatch]:
        ...


