from __future__ import annotations

from typing import Iterable, Optional, TYPE_CHECKING, Any

import torch
from torch import nn

if TYPE_CHECKING:  # avoid heavy import during tests
    from diffusers import AutoencoderKL

from src.domain.interfaces.optim_spec import OptimizerSpec
from src.domain.interfaces.types import VAEEncodeOutput


class DiffusersVAEBackbone(nn.Module):
    """
    VAE implementation backed by Diffusers' AutoencoderKL.

    Training convention:
      - encode(images, sample=True) returns latents not scaled.
      - Consumers multiply by scaling_factor before passing to UNet.
      - For decode, latents should be divided by scaling_factor internally before decoding.
    """

    def __init__(self, *, model: Any, optimizer_spec: Optional[OptimizerSpec] = None):
        super().__init__()
        self.model = model
        self._optimizer_spec = optimizer_spec or OptimizerSpec()

    # ----- Interface properties -----
    @property
    def latent_channels(self) -> int:
        return int(getattr(getattr(self.model, "config", object()), "latent_channels", 4))

    @property
    def scaling_factor(self) -> float:
        return float(getattr(getattr(self.model, "config", object()), "scaling_factor", 0.18215))

    # ----- Encode/Decode -----
    @torch.no_grad()
    def encode(self, images: torch.Tensor, sample: bool = True) -> VAEEncodeOutput:
        posterior = self.model.encode(images).latent_dist
        latents = posterior.sample() if sample else posterior.mean
        mean = getattr(posterior, "mean", None)
        if mean is None:
            mean = torch.zeros_like(latents)
        # Compute logvar if available, otherwise best-effort
        if hasattr(posterior, "logvar") and getattr(posterior, "logvar") is not None:
            logvar = posterior.logvar
        elif hasattr(posterior, "std") and getattr(posterior, "std") is not None:
            std = posterior.std
            logvar = (std * std + 1e-6).log()
        elif hasattr(posterior, "var") and getattr(posterior, "var") is not None:
            var = posterior.var
            logvar = (var + 1e-6).log()
        else:
            logvar = torch.zeros_like(latents)
        return VAEEncodeOutput(latents=latents, mean=mean, logvar=logvar)

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        # Expect latents scaled for UNet; divide by scaling to decode
        unscaled = latents / self.scaling_factor
        out = self.model.decode(unscaled)
        images = getattr(out, "sample", out)
        return images

    def optimizer_spec(self) -> OptimizerSpec:
        return self._optimizer_spec


