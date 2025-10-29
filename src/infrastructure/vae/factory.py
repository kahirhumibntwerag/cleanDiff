from __future__ import annotations

from typing import Optional, Sequence

import torch

from diffusers import AutoencoderKL

from .diffusers_vae import DiffusersVAEBackbone
from src.domain.interfaces.optim_spec import OptimizerSpec


def create_diffusers_vae_from_pretrained(
    model_name_or_path: str,
    *,
    subfolder: Optional[str] = None,
    revision: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
    variant: Optional[str] = None,
    use_safetensors: bool = True,
    cache_dir: Optional[str] = None,
    local_files_only: bool = False,
    optimizer_spec: Optional[OptimizerSpec] = None,
) -> DiffusersVAEBackbone:
    model = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path,
        subfolder=subfolder,
        revision=revision,
        torch_dtype=torch_dtype,
        variant=variant,
        use_safetensors=use_safetensors,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    return DiffusersVAEBackbone(model=model, optimizer_spec=optimizer_spec)


def create_diffusers_vae(
    *,
    in_channels: int = 3,
    out_channels: int = 3,
    latent_channels: int = 4,
    block_out_channels: Sequence[int] = (128, 256, 256),
    down_block_types: Sequence[str] = ("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"),
    up_block_types: Sequence[str] = ("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"),
    layers_per_block: int = 2,
    sample_size: Optional[int] = None,
    scaling_factor: Optional[float] = 0.18215,
    norm_num_groups: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
    optimizer_spec: Optional[OptimizerSpec] = None,
) -> DiffusersVAEBackbone:
    """
    Create a lightweight AutoencoderKL with configurable blocks and wrap it.
    Note: For production, prefer from_pretrained unless you know the config you want.
    """
    kwargs = {
        "in_channels": in_channels,
        "out_channels": out_channels,
        "latent_channels": latent_channels,
        "block_out_channels": tuple(block_out_channels),
        "down_block_types": tuple(down_block_types),
        "up_block_types": tuple(up_block_types),
        "layers_per_block": layers_per_block,
    }
    if sample_size is not None:
        kwargs["sample_size"] = sample_size

    # Some diffusers versions require `norm_num_groups`; others reject it.
    # Prefer passing a safe default (32) and fall back to no-arg if unsupported.
    try:
        desired_norm = 32 if norm_num_groups is None else norm_num_groups
        model = AutoencoderKL(**({**kwargs, "norm_num_groups": desired_norm}))
    except TypeError as e:
        if "norm_num_groups" in str(e):
            model = AutoencoderKL(**kwargs)
        else:
            raise
    if scaling_factor is not None:
        # Update config to reflect scaling factor convention used with UNet
        if hasattr(model, "config"):
            setattr(model.config, "scaling_factor", float(scaling_factor))
    if dtype is not None:
        model = model.to(dtype=dtype)
    return DiffusersVAEBackbone(model=model, optimizer_spec=optimizer_spec)


