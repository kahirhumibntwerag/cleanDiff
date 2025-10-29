from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch

from .types import PredictionType, SchedulerStepOutput


@runtime_checkable
class Scheduler(Protocol):
    """
    Diffusion scheduler controlling the sampling trajectory.
    """

    @property
    def num_train_timesteps(self) -> int:
        ...

    @property
    def init_noise_sigma(self) -> float:
        """
        Sigma of initial noise distribution used to normalize latents at t=T.
        """
        ...

    @property
    def prediction_type(self) -> PredictionType:
        """
        Indicates what the UNet predicts: epsilon, v_prediction, or sample.
        """
        ...

    def set_timesteps(self, num_inference_steps: int, device: torch.device) -> torch.Tensor:
        """
        Precompute and store the inference timesteps.
        Returns: 1-D long Tensor of shape [num_inference_steps] on the provided device.
        """
        ...

    def scale_model_input(self, sample: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Optionally scale/normalize input latents before UNet (e.g., Karras sigmas).
        """
        ...

    def step(
        self,
        model_output: torch.Tensor,     # UNet output aligned with prediction_type
        timestep: torch.Tensor,         # current t
        sample: torch.Tensor,           # current x_t
        return_dict: bool = True,
    ) -> SchedulerStepOutput:
        """
        Compute x_{t-1} and optionally x0 estimate.
        Implementations MUST return SchedulerStepOutput.
        """
        ...

    def add_noise(
        self,
        original_samples: torch.Tensor,  # x0
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        q(x_t | x0) to generate noisy latents for training.
        """
        ...

    def get_velocity(
        self,
        sample: torch.Tensor,     # x_t
        noise: torch.Tensor,      # ε
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """
        Optional helper for v-pred training targets.
        """
        ...


