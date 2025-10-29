from __future__ import annotations

from typing import Optional, Protocol, Tuple, runtime_checkable

import torch

from .encoder import Encoder
from .scheduler import Scheduler
from .unet import UNetBackbone
from .vae import VAE


@runtime_checkable
class Sampler(Protocol):
    """
    Sampler interface that drives the denoising loop to generate images from latents.
    Implementations should use the provided UNet, Scheduler, and VAE to sample.
    """

    @torch.no_grad()
    def sample(
        self,
        *,
        unet: UNetBackbone,
        scheduler: Scheduler,
        vae: VAE,
        encoder_hidden_states: Optional[torch.Tensor],
        num_inference_steps: int,
        latents_shape: Tuple[int, int, int, int],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        ...


