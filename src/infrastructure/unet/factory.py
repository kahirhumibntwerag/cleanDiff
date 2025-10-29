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
        heads_tuple = tuple(num_attention_heads) if isinstance(num_attention_heads, (list, tuple)) else num_attention_heads
        kwargs["num_attention_heads"] = heads_tuple
    else:
        # Derive per-block heads based on block_out_channels to ensure divisibility
        per_block_heads = []
        for out_ch in block_out_channels:
            # Aim for ~64 dims per head, fallback to a divisor of out_ch
            target = max(1, int(out_ch // 64))
            h = max(1, target)
            while out_ch % h != 0 and h > 1:
                h -= 1
            per_block_heads.append(h)
        kwargs["num_attention_heads"] = tuple(per_block_heads)

    # Ensure attention_head_dim matches per-block so that inner_dim = out_ch
    if attention_head_dim is None:
        per_block_head_dim = []
        heads = kwargs["num_attention_heads"]
        heads_list = list(heads) if isinstance(heads, (list, tuple)) else [int(heads)] * len(block_out_channels)
        for out_ch, h in zip(block_out_channels, heads_list):
            per_block_head_dim.append(int(out_ch // max(1, int(h))))
        kwargs["attention_head_dim"] = tuple(per_block_head_dim)

    # Some diffusers versions require `norm_num_groups`; others reject it.
    # Some older versions also reject `attention_head_dim`.
    desired_norm = 32 if norm_num_groups is None else norm_num_groups
    try:
        model = UNetSpatioTemporalConditionModel(**({**kwargs, "norm_num_groups": desired_norm}))
    except TypeError as e:
        msg = str(e)
        if "attention_head_dim" in msg:
            # Retry without attention_head_dim and adjust block_out_channels to multiples of 88 * heads
            k2 = dict(kwargs)
            k2.pop("attention_head_dim", None)
            heads = k2.get("num_attention_heads", ())
            heads_list = list(heads) if isinstance(heads, (list, tuple)) else [int(heads)] * len(block_out_channels)
            new_blocks = []
            for out_ch, h in zip(block_out_channels, heads_list):
                h_i = max(1, int(h))
                # Ensure heads is a multiple of 4 so that (heads*88) is divisible by 32 for GroupNorm
                if h_i % 4 != 0:
                    h_i = ((h_i // 4) + 1) * 4
                new_blocks.append(h_i * 88)
            k2["block_out_channels"] = tuple(new_blocks)
            try:
                model = UNetSpatioTemporalConditionModel(**({**k2, "norm_num_groups": desired_norm}))
            except TypeError as e3:
                if "norm_num_groups" in str(e3):
                    try:
                        model = UNetSpatioTemporalConditionModel(**k2)
                    except TypeError as e4:
                        if "num_groups" in str(e4) or "GroupNorm" in str(e4) or "%: 'int' and 'NoneType'" in str(e4):
                            st_kwargs = dict(k2)
                            st_kwargs["down_block_types"] = ("DownBlockSpatioTemporal",) * len(tuple(down_block_types))
                            st_kwargs["up_block_types"] = ("UpBlockSpatioTemporal",) * len(tuple(up_block_types))
                            model = UNetSpatioTemporalConditionModel(**st_kwargs)
                        else:
                            raise
                else:
                    raise
        elif "norm_num_groups" in msg:
            try:
                model = UNetSpatioTemporalConditionModel(**kwargs)
            except TypeError as e2:
                # Some older variants pass `resnet_groups=None` via get_down_block causing GroupNorm errors.
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


