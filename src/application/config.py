from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Sequence

import torch

from src.domain.interfaces.optim_spec import OptimizerKind, OptimizerSpec
from src.infrastructure.optimizers.torch_builder import TorchOptimizerBuilder
from src.infrastructure.unet.factory import (
    create_diffusers_video_unet,
    create_diffusers_video_unet_from_pretrained,
)
from src.infrastructure.vae.factory import (
    create_diffusers_vae,
    create_diffusers_vae_from_pretrained,
)
from src.infrastructure.schedulers.factory import (
    create_diffusers_scheduler,
    create_diffusers_scheduler_from_pretrained,
)
from src.infrastructure.samplers import DiffusersDenoiserSampler
from src.infrastructure.encoders import create_speed_encoder


DTypeStr = Literal["float32", "float16", "bfloat16"]


def _parse_dtype(dtype: DTypeStr | None) -> Optional[torch.dtype]:
    if dtype is None:
        return None
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    return torch.float32


@dataclass
class OptimSpecConfig:
    kind: OptimizerKind = OptimizerKind.ADAMW
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    betas: Optional[tuple[float, float]] = (0.9, 0.999)
    eps: Optional[float] = None
    momentum: Optional[float] = None

    def to_spec(self) -> OptimizerSpec:
        return OptimizerSpec(
            kind=self.kind,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=self.betas,
            eps=self.eps,
            momentum=self.momentum,
        )


@dataclass
class UNetConfig:
    # Build vs pretrained
    pretrained: Optional[str] = None
    subfolder: Optional[str] = None
    revision: Optional[str] = None
    variant: Optional[str] = None
    torch_dtype: Optional[DTypeStr] = None

    # Build-time params (ignored if pretrained is provided)
    sample_size: int = 32
    in_channels: int = 4
    out_channels: int = 4
    block_out_channels: Sequence[int] = (128, 256, 256)
    down_block_types: Sequence[str] = ("DownBlock3D", "DownBlock3D", "DownBlock3D")
    up_block_types: Sequence[str] = ("UpBlock3D", "UpBlock3D", "UpBlock3D")
    cross_attention_dim: Optional[int] = 768
    attention_head_dim: Optional[int] = None
    layers_per_block: int = 2
    norm_num_groups: Optional[int] = 32
    model_dtype: Optional[DTypeStr] = None

    optimizer: OptimSpecConfig = field(default_factory=OptimSpecConfig)


@dataclass
class VAEConfig:
    # Build vs pretrained
    pretrained: Optional[str] = None
    subfolder: Optional[str] = None
    revision: Optional[str] = None
    variant: Optional[str] = None
    torch_dtype: Optional[DTypeStr] = None

    # Build-time params
    in_channels: int = 3
    out_channels: int = 3
    latent_channels: int = 4
    block_out_channels: Sequence[int] = (128, 256, 256)
    down_block_types: Sequence[str] = ("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D")
    up_block_types: Sequence[str] = ("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D")
    layers_per_block: int = 2
    sample_size: Optional[int] = None
    scaling_factor: Optional[float] = 0.18215
    norm_num_groups: Optional[int] = 32
    model_dtype: Optional[DTypeStr] = None

    optimizer: OptimSpecConfig = field(default_factory=OptimSpecConfig)


@dataclass
class SchedulerConfig:
    # Choose scheduler kind; if pretrained is set, also specify kind to load matching subfolder
    kind: Literal[
        "ddpm",
        "ddim",
        "euler",
        "euler_ancestral",
        "heun",
        "dpmpp_2m",
        "unipc",
    ] = "euler"

    pretrained: Optional[str] = None
    subfolder: Optional[str] = None
    revision: Optional[str] = None
    variant: Optional[str] = None
    torch_dtype: Optional[DTypeStr] = None

    # Free-form kwargs forwarded to scheduler constructor
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SamplerConfig:
    # Placeholder for sampler-specific params; our simple sampler has none
    pass


@dataclass
class TrainingConfig:
    seed: int = 42
    device: Optional[str] = None  # e.g., "cuda", "cpu"
    precision: DTypeStr = "float32"
    batch_size: int = 2
    num_workers: int = 4
    max_epochs: int = 1
    gradient_accumulation_steps: int = 1
    image_size: int = 256
    num_inference_steps: int = 25


@dataclass
class SystemConfig:
    unet: UNetConfig = field(default_factory=UNetConfig)
    vae: VAEConfig = field(default_factory=VAEConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    # Encoder (optional; if provided, size should match UNet cross_attention_dim)
    encoder_enabled: bool = False
    encoder_hidden_size: Optional[int] = None
    encoder_speed_scale: float = 1000.0
    encoder_num_fourier_frequencies: int = 16
    encoder_max_fourier_frequency: float = 512.0
    encoder_mlp_layers: int = 2
    encoder_mlp_width: int = 256
    encoder_optimizer: OptimSpecConfig = field(default_factory=OptimSpecConfig)


def build_components(cfg: SystemConfig):
    # Optimizer builder (torch)
    optimizer_builder = TorchOptimizerBuilder()

    # VAE
    if cfg.vae.pretrained is not None:
        vae = create_diffusers_vae_from_pretrained(
            cfg.vae.pretrained,
            subfolder=cfg.vae.subfolder,
            revision=cfg.vae.revision,
            torch_dtype=_parse_dtype(cfg.vae.torch_dtype),
            variant=cfg.vae.variant,
            optimizer_spec=cfg.vae.optimizer.to_spec(),
        )
    else:
        vae = create_diffusers_vae(
            in_channels=cfg.vae.in_channels,
            out_channels=cfg.vae.out_channels,
            latent_channels=cfg.vae.latent_channels,
            block_out_channels=cfg.vae.block_out_channels,
            down_block_types=cfg.vae.down_block_types,
            up_block_types=cfg.vae.up_block_types,
            layers_per_block=cfg.vae.layers_per_block,
            sample_size=cfg.vae.sample_size,
            scaling_factor=cfg.vae.scaling_factor,
            norm_num_groups=cfg.vae.norm_num_groups,
            dtype=_parse_dtype(cfg.vae.model_dtype),
            optimizer_spec=cfg.vae.optimizer.to_spec(),
        )

    # UNet
    if cfg.unet.pretrained is not None:
        unet = create_diffusers_video_unet_from_pretrained(
            cfg.unet.pretrained,
            subfolder=cfg.unet.subfolder,
            revision=cfg.unet.revision,
            torch_dtype=_parse_dtype(cfg.unet.torch_dtype),
            variant=cfg.unet.variant,
            optimizer_spec=cfg.unet.optimizer.to_spec(),
        )
    else:
        unet = create_diffusers_video_unet(
            sample_size=cfg.unet.sample_size,
            in_channels=cfg.unet.in_channels,
            out_channels=cfg.unet.out_channels,
            block_out_channels=cfg.unet.block_out_channels,
            down_block_types=cfg.unet.down_block_types,
            up_block_types=cfg.unet.up_block_types,
            cross_attention_dim=cfg.unet.cross_attention_dim,
            attention_head_dim=cfg.unet.attention_head_dim,
            layers_per_block=cfg.unet.layers_per_block,
            norm_num_groups=cfg.unet.norm_num_groups,
            dtype=_parse_dtype(cfg.unet.model_dtype),
            optimizer_spec=cfg.unet.optimizer.to_spec(),
        )

    # Scheduler
    if cfg.scheduler.pretrained is not None:
        scheduler = create_diffusers_scheduler_from_pretrained(
            cfg.scheduler.pretrained,
            kind=cfg.scheduler.kind,
            subfolder=cfg.scheduler.subfolder,
            revision=cfg.scheduler.revision,
            torch_dtype=_parse_dtype(cfg.scheduler.torch_dtype),
            variant=cfg.scheduler.variant,
        )
    else:
        scheduler = create_diffusers_scheduler(kind=cfg.scheduler.kind, **cfg.scheduler.kwargs)

    # Sampler
    sampler = DiffusersDenoiserSampler()

    # Optional encoder (ensure hidden size matches UNet cross-attention dim when available)
    encoder = None
    if cfg.encoder_enabled:
        hidden_size = cfg.encoder_hidden_size
        if hidden_size is None:
            # try to infer from UNet when possible
            hidden_size = getattr(components_unet := unet, "cross_attention_dim", None) or cfg.unet.cross_attention_dim or 768  # type: ignore[attr-defined]
        encoder = create_speed_encoder(
            hidden_size=hidden_size,
            speed_scale=cfg.encoder_speed_scale,
            num_fourier_frequencies=cfg.encoder_num_fourier_frequencies,
            max_fourier_frequency=cfg.encoder_max_fourier_frequency,
            mlp_layers=cfg.encoder_mlp_layers,
            mlp_width=cfg.encoder_mlp_width,
            optimizer_spec=cfg.encoder_optimizer.to_spec(),
        )

    return {
        "vae": vae,
        "unet": unet,
        "scheduler": scheduler,
        "sampler": sampler,
        "optimizer_builder": optimizer_builder,
        "encoder": encoder,
    }


def default_config() -> SystemConfig:
    return SystemConfig()


