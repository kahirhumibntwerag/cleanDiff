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
        permuted_for_model = False
        # Heuristic: real diffusers models live under the 'diffusers.' module
        model_module = getattr(self.model.__class__, "__module__", "")
        is_diffusers_model = isinstance(model_module, str) and model_module.startswith("diffusers.")
        if latents.ndim == 4:
            if is_diffusers_model:
                # [B, C, H, W] -> [B, F=1, C, H, W] for diffusers
                latents = latents.unsqueeze(1)
            else:
                # [B, C, H, W] -> [B, C, F=1, H, W] for generic/fake models
                latents = latents.unsqueeze(2)
            squeezed = True
        elif latents.ndim == 5:
            if is_diffusers_model:
                # Assume [B, C, F, H, W] and permute to [B, F, C, H, W] for diffusers 3D UNet
                latents = latents.permute(0, 2, 1, 3, 4)
                permuted_for_model = True
            else:
                # Keep as-is for generic/fake models expecting [B, C, F, H, W]
                pass
        else:
            raise ValueError(f"Expected latents with 4 or 5 dims, got shape {tuple(latents.shape)}")

        # Diffusers expects argument name `timestep` (int or tensor)
        call_kwargs: Dict[str, Any] = {
            "sample": latents,
            "timestep": timesteps,
            "return_dict": False,
        }
        if encoder_hidden_states is not None:
            call_kwargs["encoder_hidden_states"] = encoder_hidden_states
        # Older diffusers may not accept `added_cond_kwargs`; only pass if provided
        if added_cond_kwargs is not None:
            call_kwargs["added_cond_kwargs"] = added_cond_kwargs

        try:
            out = self.model(**call_kwargs)
        except TypeError as e:
            # Older diffusers requires `added_time_ids` positional arg; synthesize zeros
            if "added_time_ids" in str(e):
                batch_size = latents.shape[0]
                # Most older UNet variants default to addition_time_embed_dim=256 and
                # projection_class_embeddings_input_dim=768 -> require 3 ids (3*256=768)
                added_time_ids = torch.zeros(batch_size, 3, device=latents.device, dtype=latents.dtype)
                call_kwargs["added_time_ids"] = added_time_ids
                out = self.model(**call_kwargs)
            else:
                raise

        # return is tuple (sample,) when return_dict=False
        if isinstance(out, (tuple, list)):
            pred = out[0]
        else:
            pred = out

        # For diffusers: model returns [B, F, C, H, W]
        if permuted_for_model:
            # [B, F, C, H, W] -> [B, C, F, H, W]
            pred = pred.permute(0, 2, 1, 3, 4)
        if squeezed:
            if is_diffusers_model:
                # [B, 1, C, H, W] -> [B, C, H, W]
                pred = pred.squeeze(1)
            else:
                # [B, C, 1, H, W] -> [B, C, H, W]
                pred = pred.squeeze(2)
        return pred

    # ----- Training hooks -----
    def optimizer_spec(self) -> OptimizerSpec:
        return self._optimizer_spec


