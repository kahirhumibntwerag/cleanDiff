from __future__ import annotations

from typing import Optional, Sequence

import torch

from diffusers import UNetSpatioTemporalConditionModel

from .diffusers_video_unet import DiffusersVideoUNetBackbone
from src.domain.interfaces.optim_spec import OptimizerSpec


def create_diffusers_video_unet(
    *,
    sample_size: int,
    in_channels: int = 4,
    out_channels: int = 4,
    block_out_channels: Sequence[int] = (128, 256, 256),
    down_block_types: Sequence[str] = ("DownBlock3D", "DownBlock3D", "DownBlock3D"),
    up_block_types: Sequence[str] = ("UpBlock3D", "UpBlock3D", "UpBlock3D"),
    cross_attention_dim: int | None = 768,
    attention_head_dim: int | None = None,
    layers_per_block: int = 2,
    norm_num_groups: int | None = 32,
    dtype: torch.dtype | None = None,
    optimizer_spec: OptimizerSpec | None = None,
) -> DiffusersVideoUNetBackbone:
    """
    Create a Diffusers UNetSpatioTemporalConditionModel configured for video (3D) and wrap it to conform to UNetBackbone.

    Defaults reflect a lightweight configuration suitable for small latent sizes.
    """

    kwargs = {
        "sample_size": sample_size,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "block_out_channels": tuple(block_out_channels),
        "down_block_types": tuple(down_block_types),
        "up_block_types": tuple(up_block_types),
        "layers_per_block": layers_per_block,
    }
    if cross_attention_dim is not None:
        kwargs["cross_attention_dim"] = cross_attention_dim
    if attention_head_dim is not None:
        kwargs["attention_head_dim"] = attention_head_dim
    if norm_num_groups is not None:
        kwargs["norm_num_groups"] = norm_num_groups

    model = UNetSpatioTemporalConditionModel(**kwargs)
    if dtype is not None:
        model = model.to(dtype=dtype)
    return DiffusersVideoUNetBackbone(model=model, optimizer_spec=optimizer_spec)


def create_diffusers_video_unet_from_pretrained(
    model_name_or_path: str,
    *,
    subfolder: Optional[str] = None,
    revision: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
    variant: Optional[str] = None,
    use_safetensors: bool = True,
    cache_dir: Optional[str] = None,
    local_files_only: bool = False,
    optimizer_spec: OptimizerSpec | None = None,
) -> DiffusersVideoUNetBackbone:
    """
    Load a UNetSpatioTemporalConditionModel from the Hugging Face Hub or local path and wrap it.
    """
    model = UNetSpatioTemporalConditionModel.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path,
        subfolder=subfolder,
        revision=revision,
        torch_dtype=torch_dtype,
        variant=variant,
        use_safetensors=use_safetensors,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    return DiffusersVideoUNetBackbone(model=model, optimizer_spec=optimizer_spec)


