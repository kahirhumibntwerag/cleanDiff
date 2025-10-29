from __future__ import annotations

from typing import Iterable, Protocol, runtime_checkable

import torch

from .types import VAEEncodeOutput
from .optim_spec import OptimizerSpec


@runtime_checkable
class VAE(Protocol):
    """
    Variational Autoencoder interface for latent diffusion.
    """

    @property
    def latent_channels(self) -> int:
        ...

    @property
    def scaling_factor(self) -> float:
        """
        Factor to scale latents between VAE and UNet spaces.
        E.g., Stable Diffusion uses ~0.18215.
        """
        ...

    def encode(self, images: torch.Tensor, sample: bool = True) -> VAEEncodeOutput:
        """
        images: [B, 3, H, W] in [-1, 1] or [0, 1] depending on implementation.
        returns: latents (optionally sampled), plus mean/logvar.
        """
        ...

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        latents: [B, C_latent, H_latent, W_latent]
        returns: images [B, 3, H, W] in [-1, 1] or [0, 1].
        """
        ...

    def to(self, device: torch.device, dtype: torch.dtype | None = None) -> "VAE":
        ...

    # Training-related hooks
    def parameters(self) -> Iterable[torch.nn.Parameter]:
        ...

    def optimizer_spec(self) -> OptimizerSpec:
        ...


