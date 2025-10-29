from __future__ import annotations

from typing import Literal, Optional

import torch

from diffusers import (
    DDPMScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    HeunDiscreteScheduler,
    DPMSolverMultistepScheduler,
    UniPCMultistepScheduler,
)

from .diffusers_scheduler import DiffusersSchedulerAdapter


SchedulerKind = Literal[
    "ddpm",
    "ddim",
    "euler",
    "euler_ancestral",
    "heun",
    "dpmpp_2m",
    "unipc",
]


def create_diffusers_scheduler(kind: SchedulerKind = "euler", **kwargs) -> DiffusersSchedulerAdapter:
    if kind == "ddpm":
        return DiffusersSchedulerAdapter(DDPMScheduler(**kwargs))
    if kind == "ddim":
        return DiffusersSchedulerAdapter(DDIMScheduler(**kwargs))
    if kind == "euler":
        return DiffusersSchedulerAdapter(EulerDiscreteScheduler(**kwargs))
    if kind == "euler_ancestral":
        return DiffusersSchedulerAdapter(EulerAncestralDiscreteScheduler(**kwargs))
    if kind == "heun":
        return DiffusersSchedulerAdapter(HeunDiscreteScheduler(**kwargs))
    if kind == "dpmpp_2m":
        return DiffusersSchedulerAdapter(DPMSolverMultistepScheduler(**kwargs))
    if kind == "unipc":
        return DiffusersSchedulerAdapter(UniPCMultistepScheduler(**kwargs))
    raise ValueError(f"Unsupported scheduler kind: {kind}")


def create_diffusers_scheduler_from_pretrained(
    model_name_or_path: str,
    *,
    kind: SchedulerKind = "euler",
    subfolder: Optional[str] = None,
    revision: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
    variant: Optional[str] = None,
    use_safetensors: bool = True,
    cache_dir: Optional[str] = None,
    local_files_only: bool = False,
) -> DiffusersSchedulerAdapter:
    cls_map = {
        "ddpm": DDPMScheduler,
        "ddim": DDIMScheduler,
        "euler": EulerDiscreteScheduler,
        "euler_ancestral": EulerAncestralDiscreteScheduler,
        "heun": HeunDiscreteScheduler,
        "dpmpp_2m": DPMSolverMultistepScheduler,
        "unipc": UniPCMultistepScheduler,
    }
    cls = cls_map[kind]
    scheduler = cls.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path,
        subfolder=subfolder,
        revision=revision,
        torch_dtype=torch_dtype,
        variant=variant,
        use_safetensors=use_safetensors,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    return DiffusersSchedulerAdapter(scheduler)


