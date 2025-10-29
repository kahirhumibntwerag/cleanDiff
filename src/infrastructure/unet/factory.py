from __future__ import annotations

from typing import Optional, Sequence, Union

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
    num_attention_heads: Union[int, Sequence[int], None] = None,
    layers_per_block: int = 2,
    norm_num_groups: int | None = None,
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
    # Ensure num_attention_heads length matches down_block_types for older diffusers versions
    if num_attention_heads is not None:
        kwargs["num_attention_heads"] = tuple(num_attention_heads) if isinstance(num_attention_heads, (list, tuple)) else num_attention_heads
    elif cross_attention_dim is not None:
        # Derive a reasonable default: 64-d head size => heads = cross_dim // 64
        # Repeat per down block to satisfy diffusers validation.
        derived_heads = max(1, int(cross_attention_dim // 64))
        kwargs["num_attention_heads"] = tuple([derived_heads] * len(down_block_types))

    # Some diffusers versions require `norm_num_groups`; others reject it.
    # Prefer passing a safe default (32) and fall back to no-arg if unsupported.
    try:
        desired_norm = 32 if norm_num_groups is None else norm_num_groups
        model = UNetSpatioTemporalConditionModel(**({**kwargs, "norm_num_groups": desired_norm}))
    except TypeError as e:
        # Older versions: `norm_num_groups` not accepted; retry without it
        if "norm_num_groups" in str(e):
            try:
                model = UNetSpatioTemporalConditionModel(**kwargs)
            except TypeError as e2:
                # Some older variants pass `resnet_groups=None` via get_down_block causing GroupNorm errors.
                # Fallback to spatio-temporal blocks which don't rely on `resnet_groups` kw.
                if "num_groups" in str(e2) or "GroupNorm" in str(e2) or "%: 'int' and 'NoneType'" in str(e2):
                    st_kwargs = dict(kwargs)
                    st_kwargs["down_block_types"] = ("DownBlockSpatioTemporal",) * len(tuple(down_block_types))
                    st_kwargs["up_block_types"] = ("UpBlockSpatioTemporal",) * len(tuple(up_block_types))
                    model = UNetSpatioTemporalConditionModel(**st_kwargs)
                else:
                    raise
        else:
            raise
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


