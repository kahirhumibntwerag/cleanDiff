from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, Union, runtime_checkable, Iterable

import torch
from .optim_spec import OptimizerSpec


@runtime_checkable
class Encoder(Protocol):
    """
    Generic conditioning encoder interface that produces encoder hidden states for the UNet.

    Implementations might be a text encoder (e.g., Transformer), CLIP text encoder,
    or any module that maps input features to conditioning states.
    """

    @property
    def hidden_size(self) -> int:
        """
        Dimensionality D of the returned hidden states.
        """
        ...

    def forward(
        self,
        inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        *,
        return_dict: bool = False,
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        """
        Compute encoder hidden states.

        Args:
            inputs: Either a tensor (e.g., token embeddings or raw features) or a dict
                    of named tensors (e.g., {"input_ids", "attention_mask", ...}).
            return_dict: If True, return a dict with key 'encoder_hidden_states'; otherwise,
                         return the hidden states tensor directly.

        Returns:
            - Tensor of shape [B, S, D] (sequence) or [B, D] (pooled), or
            - Dict with key 'encoder_hidden_states' -> tensor as above.
        """
        ...

    def __call__(
        self,
        inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        *,
        return_dict: bool = False,
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        ...

    def to(self, device: torch.device, dtype: Optional[torch.dtype] = None) -> "Encoder":
        ...

    # Training-related hooks
    def parameters(self) -> Iterable[torch.nn.Parameter]:
        ...

    def optimizer_spec(self) -> OptimizerSpec:
        ...


