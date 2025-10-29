from __future__ import annotations

import pytest
import torch

from src.infrastructure.schedulers.factory import create_diffusers_scheduler
from src.infrastructure.schedulers.diffusers_scheduler import DiffusersSchedulerAdapter


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


def test_scale_model_input_per_sample_fallback_when_vector_not_supported(monkeypatch: pytest.MonkeyPatch) -> None:
    class _S:
        def scale_model_input(self, sample: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
            # Only supports scalar timestep; multiplies by (1 + timestep*0.01)
            if isinstance(timestep, torch.Tensor) and timestep.ndim > 0:
                raise TypeError("vector timestep not supported")
            factor = 1.0 + float(timestep.item()) * 0.01
            return sample * factor

    sch = DiffusersSchedulerAdapter(_S())
    b, c, h, w = 3, 1, 2, 2
    x = torch.ones(b, c, h, w)
    t = torch.tensor([0, 10, 20], dtype=torch.long)
    y = sch.scale_model_input(sample=x, timestep=t)
    # Expect per-sample scaling factors 1.0, 1.1, 1.2
    assert torch.allclose(y[0], x[0] * 1.0)
    assert torch.allclose(y[1], x[1] * 1.1)
    assert torch.allclose(y[2], x[2] * 1.2)


