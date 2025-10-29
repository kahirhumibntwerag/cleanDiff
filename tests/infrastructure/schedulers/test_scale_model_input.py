from __future__ import annotations

import pytest
import torch

from src.infrastructure.schedulers.factory import create_diffusers_scheduler


@pytest.mark.parametrize("kind", ["euler"])  # target the scheduler that requires scalar timestep
def test_scale_model_input_accepts_batched_timestep(kind: str) -> None:
    # Create scheduler adapter
    sch = create_diffusers_scheduler(kind=kind)

    # Fake batch of latents and batched timesteps
    b, c, h, w = 4, 4, 8, 8
    x = torch.randn(b, c, h, w)
    t = torch.randint(0, sch.num_train_timesteps, (b,), dtype=torch.long)

    # Should not raise and should preserve the shape
    y = sch.scale_model_input(sample=x, timestep=t)
    assert isinstance(y, torch.Tensor)
    assert y.shape == x.shape


