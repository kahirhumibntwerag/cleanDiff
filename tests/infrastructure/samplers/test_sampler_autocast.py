from __future__ import annotations

import torch
import pytest

from src.infrastructure.samplers.diffusers_sampler import DiffusersDenoiserSampler
from tests.fakes import FakeScheduler, FakeVAE


class _ToyUNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Float32 weights to simulate typical model params
        self.lin = torch.nn.Linear(4, 4).to(dtype=torch.float32)

    def forward(self, latents: torch.Tensor, timesteps: torch.Tensor, encoder_hidden_states=None, added_cond_kwargs=None) -> torch.Tensor:  # type: ignore[override]
        b, c, h, w = latents.shape
        x = latents.reshape(b, c, h * w)[..., :4]
        y = self.lin(x)  # will raise without autocast if x is bf16/half and weights are float32
        s = y.mean()
        return latents + s.to(dtype=latents.dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_sampler_autocast_prevents_dtype_mismatch_on_cpu(dtype: torch.dtype) -> None:
    device = torch.device("cpu")
    unet = _ToyUNet()
    scheduler = FakeScheduler()
    vae = FakeVAE(latent_channels=4)

    sampler = DiffusersDenoiserSampler()

    imgs = sampler.sample(
        unet=unet,
        scheduler=scheduler,
        vae=vae,
        encoder_hidden_states=None,
        num_inference_steps=2,
        latents_shape=(2, 4, 2, 2),
        device=device,
        dtype=dtype,
    )
    assert isinstance(imgs, torch.Tensor)
    assert imgs.ndim == 4

