from __future__ import annotations

from typing import Dict, Iterable, Optional, Union

import math
import torch
from torch import nn

from src.domain.interfaces.optim_spec import OptimizerSpec


def _fourier_features(x: torch.Tensor, num_frequencies: int, max_freq: float) -> torch.Tensor:
    """
    Map a scalar tensor x -> [sin(2^k*pi*x), cos(2^k*pi*x)] features.
    x shape: [B] or [B, 1]. Returns [B, 2 * num_frequencies].
    """
    if x.ndim == 2 and x.size(-1) == 1:
        x = x.squeeze(-1)
    B = x.size(0)
    # Frequencies geometrically spaced up to max_freq
    freqs = torch.logspace(0, math.log2(max_freq + 1e-6), steps=num_frequencies, base=2.0, device=x.device, dtype=x.dtype)
    phases = x[:, None] * freqs[None, :] * math.pi
    return torch.cat([torch.sin(phases), torch.cos(phases)], dim=-1)


class SpeedEncoder(nn.Module):
    """
    Encodes scalar speeds (e.g., values in the hundreds) into encoder hidden states matching
    the UNet cross-attention dimension. Outputs shape [B, 1, hidden_size].
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        speed_scale: float = 1000.0,
        num_fourier_frequencies: int = 16,
        max_fourier_frequency: float = 512.0,
        mlp_layers: int = 2,
        mlp_width: int = 256,
        optimizer_spec: Optional[OptimizerSpec] = None,
    ) -> None:
        super().__init__()
        self._hidden_size = int(hidden_size)
        self.speed_scale = float(speed_scale)
        self.num_fourier_frequencies = int(num_fourier_frequencies)
        self.max_fourier_frequency = float(max_fourier_frequency)
        self._optimizer_spec = optimizer_spec or OptimizerSpec()

        in_dim = 2 * self.num_fourier_frequencies + 1  # + raw normalized speed
        layers: list[nn.Module] = []
        last = in_dim
        for _ in range(max(mlp_layers - 1, 0)):
            layers += [nn.Linear(last, mlp_width), nn.SiLU()]
            last = mlp_width
        layers.append(nn.Linear(last, self._hidden_size))
        self.mlp = nn.Sequential(*layers)

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    def forward(
        self,
        inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        *,
        return_dict: bool = False,
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        # Extract speeds as tensor [B] or [B,1]
        if isinstance(inputs, dict):
            speeds = inputs.get("speed")
            if speeds is None:
                speeds = inputs.get("speeds")
            if speeds is None:
                raise KeyError("SpeedEncoder expects 'speed' or 'speeds' in inputs dict")
        else:
            speeds = inputs

        if speeds.ndim == 1:
            speeds = speeds[:, None]
        # Normalize to roughly [0, 1] range for values in the hundreds
        norm = (speeds / self.speed_scale).clamp_min(0.0)
        ff = _fourier_features(norm, self.num_fourier_frequencies, self.max_fourier_frequency)
        feats = torch.cat([norm, ff], dim=-1)
        # Ensure matmul dtype matches module weights to avoid Half/Float mismatch under mixed precision
        orig_dtype = feats.dtype
        try:
            param_dtype = next(self.mlp.parameters()).dtype
        except StopIteration:
            param_dtype = orig_dtype
        feats = feats.to(dtype=param_dtype)
        hs = self.mlp(feats)  # [B, D]
        # Return to original requested dtype for downstream modules
        if hs.dtype != orig_dtype:
            hs = hs.to(dtype=orig_dtype)
        hs = hs[:, None, :]   # [B, 1, D]

        if return_dict:
            return {"encoder_hidden_states": hs}
        return hs

    def to(self, device: torch.device, dtype: Optional[torch.dtype] = None) -> "SpeedEncoder":
        return super().to(device=device, dtype=dtype)  # type: ignore[return-value]

    # Training hooks
    def parameters(self) -> Iterable[torch.nn.Parameter]:  # type: ignore[override]
        return super().parameters()

    def optimizer_spec(self) -> OptimizerSpec:
        return self._optimizer_spec


