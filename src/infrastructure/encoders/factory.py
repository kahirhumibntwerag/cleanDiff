from __future__ import annotations

from typing import Optional

from src.domain.interfaces.optim_spec import OptimizerSpec
from .speed_encoder import SpeedEncoder


def create_speed_encoder(
    *,
    hidden_size: int,
    speed_scale: float = 1000.0,
    num_fourier_frequencies: int = 16,
    max_fourier_frequency: float = 512.0,
    mlp_layers: int = 2,
    mlp_width: int = 256,
    optimizer_spec: Optional[OptimizerSpec] = None,
) -> SpeedEncoder:
    return SpeedEncoder(
        hidden_size=hidden_size,
        speed_scale=speed_scale,
        num_fourier_frequencies=num_fourier_frequencies,
        max_fourier_frequency=max_fourier_frequency,
        mlp_layers=mlp_layers,
        mlp_width=mlp_width,
        optimizer_spec=optimizer_spec,
    )


