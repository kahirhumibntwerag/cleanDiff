from __future__ import annotations

import torch

from diffusers import EulerDiscreteScheduler

from src.infrastructure.schedulers.diffusers_scheduler import DiffusersSchedulerAdapter
from src.domain.interfaces.types import PredictionType


def test_scheduler_adapter_basic_flow_on_cpu() -> None:
    base = EulerDiscreteScheduler()
    sch = DiffusersSchedulerAdapter(base)

    assert isinstance(sch.num_train_timesteps, int)
    assert isinstance(sch.init_noise_sigma, float)
    assert sch.prediction_type in {PredictionType.EPSILON, PredictionType.V_PREDICTION, PredictionType.SAMPLE}

    device = torch.device("cpu")
    timesteps = sch.set_timesteps(num_inference_steps=3, device=device)
    assert isinstance(timesteps, torch.Tensor)
    assert timesteps.device.type == "cpu"

    x = torch.randn(2, 4, 8, 8)
    t = timesteps[0]
    model_in = sch.scale_model_input(x, t)
    assert model_in.shape == x.shape

    # Model output same shape as input; step should return prev_sample with same shape
    step_out = sch.step(model_output=torch.zeros_like(x), timestep=t, sample=x)
    assert step_out.prev_sample.shape == x.shape


