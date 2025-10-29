from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from torch.utils.data import Dataset

from .types import DiffusionBatch


@runtime_checkable
class DiffusionDataset(Protocol):
    """
    Dataset protocol for diffusion training. Must return a `DiffusionBatch` per item
    after collation. Implementations typically subclass `torch.utils.data.Dataset`;
    this protocol only enforces the contract for consumers.

    Normalization rule for conditioning:
      - If the UNet is conditioned, the collated batch MUST include exactly one of
        `encoder_hidden_states` (precomputed) OR `encoder_inputs` (raw inputs).
      - Do NOT emit ad-hoc keys like `input_ids` at the top-level; put them inside
        `encoder_inputs` when needed.
    """

    def __len__(self) -> int:  # pragma: no cover - contract only
        ...

    def __getitem__(self, index: int) -> DiffusionBatch:  # single item pre-collation
        ...

    def collate_fn(self, batch: Sequence[DiffusionBatch]) -> DiffusionBatch:
        """
        Optional custom collation into a single `DiffusionBatch`.
        If not provided, a DataLoader default collate can be used as long as it
        produces a `DiffusionBatch` mapping following the normalization rule.
        """
        ...


