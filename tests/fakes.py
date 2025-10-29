from __future__ import annotations

from typing import Dict, Iterable, Optional

import torch

from src.domain.interfaces.optim_spec import OptimizerSpec
from src.domain.interfaces.types import PredictionType, SchedulerStepOutput, VAEEncodeOutput


class FakeUNet(torch.nn.Module):
    def __init__(self, in_channels: int = 4, out_channels: int = 4) -> None:
        super().__init__()
        # minimal parameter so optimizers can be constructed
        self._weight = torch.nn.Parameter(torch.zeros(1))
        self._in_channels = in_channels
        self._out_channels = out_channels

    @property
    def in_channels(self) -> int:
        return self._in_channels

    @property
    def out_channels(self) -> int:
        return self._out_channels

    def forward(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        added_cond_kwargs: Optional[Dict[str, object]] = None,
    ) -> torch.Tensor:
        # Return a tensor matching input shape regardless of conditioning
        return torch.zeros_like(latents) + self._weight

    def optimizer_spec(self) -> OptimizerSpec:
        return OptimizerSpec()


class FakeVAE(torch.nn.Module):
    def __init__(self, latent_channels: int = 4, scaling_factor: float = 0.5) -> None:
        super().__init__()
        self._latent_channels = latent_channels
        self._scaling_factor = scaling_factor

    @property
    def latent_channels(self) -> int:
        return self._latent_channels

    @property
    def scaling_factor(self) -> float:
        return self._scaling_factor

    def encode(self, images: torch.Tensor, sample: bool = True) -> VAEEncodeOutput:
        b, _, h, w = images.shape
        lat_h, lat_w = h // 8, w // 8
        latents = torch.ones(b, self.latent_channels, max(lat_h, 1), max(lat_w, 1), device=images.device)
        return VAEEncodeOutput(latents=latents, mean=torch.zeros_like(latents), logvar=torch.zeros_like(latents))

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        b, _, h, w = latents.shape
        return torch.zeros(b, 3, h * 8, w * 8, device=latents.device)

    def optimizer_spec(self) -> OptimizerSpec:
        return OptimizerSpec()


class FakeEncoder(torch.nn.Module):
    def __init__(self, hidden_size: int = 16, return_dict_even_if_false: bool = False) -> None:
        super().__init__()
        # parameter for optimizer
        self._p = torch.nn.Parameter(torch.zeros(1))
        self._hidden_size = hidden_size
        self._return_dict_even_if_false = return_dict_even_if_false

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    def forward(self, inputs: torch.Tensor | Dict[str, torch.Tensor], *, return_dict: bool = False):
        hs = torch.zeros(2, 4, self.hidden_size, device=self._p.device)
        if return_dict or self._return_dict_even_if_false:
            return {"encoder_hidden_states": hs}
        return hs

    __call__ = forward

    def optimizer_spec(self) -> OptimizerSpec:
        return OptimizerSpec()


class FakeScheduler:
    def __init__(self, prediction_type: PredictionType = PredictionType.EPSILON) -> None:
        self._prediction_type = prediction_type

    @property
    def num_train_timesteps(self) -> int:
        return 10

    @property
    def init_noise_sigma(self) -> float:
        return 1.0

    @property
    def prediction_type(self) -> PredictionType:
        return self._prediction_type

    def set_timesteps(self, num_inference_steps: int, device: torch.device) -> torch.Tensor:
        return torch.arange(num_inference_steps, device=device)

    def scale_model_input(self, sample: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        return sample

    def step(self, model_output: torch.Tensor, timestep: torch.Tensor, sample: torch.Tensor, return_dict: bool = True) -> SchedulerStepOutput:  # noqa: E501
        return SchedulerStepOutput(prev_sample=sample)

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return original_samples + noise

    def get_velocity(self, sample: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        return noise


class FakeSampler:
    @torch.no_grad()
    def sample(
        self,
        *,
        unet,
        scheduler,
        vae,
        encoder_hidden_states: torch.Tensor | None,
        num_inference_steps: int,
        latents_shape: tuple[int, int, int, int],
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        return torch.zeros(latents_shape, device=device, dtype=dtype)


class FakeOptimizerBuilder:
    def build(self, *, spec: OptimizerSpec, params: Iterable[torch.nn.Parameter]):
        return torch.optim.SGD(params, lr=spec.learning_rate)


