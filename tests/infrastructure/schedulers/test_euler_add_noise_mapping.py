from __future__ import annotations

import pytest
import torch

from src.infrastructure.schedulers.factory import create_diffusers_scheduler


pytestmark = pytest.mark.skipif(pytest.importorskip("diffusers", reason="diffusers not installed") is None, reason="diffusers not installed")  # type: ignore[truthy-bool]


def test_euler_add_noise_maps_scalar_timestep() -> None:
    sched = create_diffusers_scheduler(kind="euler")
    device = torch.device("cpu")
    # Initialize schedule with a small number of steps
    sched.set_timesteps(num_inference_steps=10, device=device)
    x = torch.randn(2, 4, 8, 8, device=device)
    noise = torch.randn_like(x)
    # Pass an integer timestep (index)
    t = torch.tensor(5, device=device, dtype=torch.long)
    out = sched.add_noise(original_samples=x, noise=noise, timesteps=t)
    assert isinstance(out, torch.Tensor)
    assert out.shape == x.shape


def test_euler_add_noise_maps_batched_timesteps() -> None:
    sched = create_diffusers_scheduler(kind="euler")
    device = torch.device("cpu")
    # Initialize schedule with a small number of steps
    sched.set_timesteps(num_inference_steps=12, device=device)
    x = torch.randn(3, 4, 8, 8, device=device)
    noise = torch.randn_like(x)
    # Batched integer timesteps, some potentially outside range to exercise clamp
    t = torch.tensor([0, 5, 20], device=device, dtype=torch.long)
    out = sched.add_noise(original_samples=x, noise=noise, timesteps=t)
    assert isinstance(out, torch.Tensor)
    assert out.shape == x.shape
