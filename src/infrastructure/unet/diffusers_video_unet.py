from __future__ import annotations

from typing import Any, Dict, Optional, TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:  # avoid importing heavy deps at runtime for tests
    from diffusers import UNetSpatioTemporalConditionModel

from src.domain.interfaces.optim_spec import OptimizerSpec
from src.domain.interfaces.unet import UNetBackbone


class DiffusersVideoUNetBackbone(nn.Module):
    """
    UNetBackbone implementation backed by Diffusers' UNetSpatioTemporalConditionModel (3D conditional UNet).

    Notes on tensor shapes:
      - This wrapper accepts 4D latents [B, C, H, W] (image) or 5D latents [B, C, F, H, W] (video).
      - If a 4D tensor is provided, a singleton frames dimension (F=1) is added and removed after the forward pass.
    """

    def __init__(self, *, model: Any, optimizer_spec: Optional[OptimizerSpec] = None) -> None:
        super().__init__()
        self.model = model
        self._optimizer_spec = optimizer_spec or OptimizerSpec()

    # ----- UNetBackbone properties -----
    @property
    def in_channels(self) -> int:
        return int(getattr(self.model.config, "in_channels", 0))

    @property
    def out_channels(self) -> int:
        return int(getattr(self.model.config, "out_channels", 0))

    @property
    def cross_attention_dim(self) -> Optional[int]:
        return getattr(self.model.config, "cross_attention_dim", None)

    # ----- UNetBackbone forward -----
    def forward(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        added_cond_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        squeezed = False
        if latents.ndim == 4:
            # [B, C, H, W] -> [B, C, F=1, H, W]
            latents = latents.unsqueeze(2)
            squeezed = True
        if latents.ndim != 5:
            raise ValueError(f"Expected latents with 4 or 5 dims, got shape {tuple(latents.shape)}")

        # Diffusers expects argument name `timestep` (int or tensor)
        out = self.model(
            sample=latents,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )

        # return is tuple (sample,) when return_dict=False
        if isinstance(out, (tuple, list)):
            pred = out[0]
        else:
            pred = out

        if squeezed:
            # [B, C, 1, H, W] -> [B, C, H, W]
            pred = pred.squeeze(2)
        return pred

    # ----- Training hooks -----
    def optimizer_spec(self) -> OptimizerSpec:
        return self._optimizer_spec


