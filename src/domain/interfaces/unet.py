from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, runtime_checkable, Iterable

import torch
from .optim_spec import OptimizerSpec


@runtime_checkable
class UNetBackbone(Protocol):
    """
    Diffusion backbone operating in latent space.
    """

    @property
    def in_channels(self) -> int:
        ...

    @property
    def out_channels(self) -> int:
        ...

    def forward(
        self,
        latents: torch.Tensor,                 # [B, C_latent, H_latent, W_latent]
        timesteps: torch.Tensor,               # [B] or scalar tensor
        encoder_hidden_states: Optional[torch.Tensor] = None,  # conditioning, e.g., text
        added_cond_kwargs: Optional[Dict[str, Any]] = None,     # e.g., time ids, guidance
    ) -> torch.Tensor:
        """
        Returns the model prediction tensor aligned with the scheduler's prediction_type
        (epsilon, v, or sample). Implementations MUST return a Tensor.
        """
        ...

    # Allow calling like a nn.Module
    def __call__(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        added_cond_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        ...


    # Training-related hooks
    def parameters(self) -> Iterable[torch.nn.Parameter]:
        ...

    def optimizer_spec(self) -> OptimizerSpec:
        ...

