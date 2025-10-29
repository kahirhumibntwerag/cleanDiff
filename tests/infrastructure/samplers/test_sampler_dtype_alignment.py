from __future__ import annotations

import torch

from src.infrastructure.samplers.diffusers_sampler import DiffusersDenoiserSampler
from tests.fakes import FakeScheduler


class _Float32ConvVAE(torch.nn.Module):
    def __init__(self, latent_channels: int = 4) -> None:
        super().__init__()
        self._latent_channels = latent_channels
        # float32 conv to simulate real VAE layer dtypes
        self.post = torch.nn.Conv2d(latent_channels, 3, kernel_size=1).to(dtype=torch.float32)

    @property
    def latent_channels(self) -> int:
        return self._latent_channels

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        # conv2d will raise a dtype mismatch if latents is half/bf16 and weights are float32
        y = self.post(latents)
        b, _, h, w = y.shape
        return torch.zeros(b, 3, h * 8, w * 8, device=y.device, dtype=y.dtype)


class _ZeroUNet(torch.nn.Module):
    def forward(self, latents: torch.Tensor, timesteps: torch.Tensor, encoder_hidden_states=None, added_cond_kwargs=None) -> torch.Tensor:  # type: ignore[override]
        return torch.zeros_like(latents)


def _run(dtype: torch.dtype, latents_shape: tuple[int, ...]) -> torch.Tensor:
    device = torch.device("cpu")
    unet = _ZeroUNet()
    scheduler = FakeScheduler()
    vae = _Float32ConvVAE(latent_channels=latents_shape[1])
    sampler = DiffusersDenoiserSampler()
    imgs = sampler.sample(
        unet=unet,
        scheduler=scheduler,
        vae=vae,
        encoder_hidden_states=None,
        num_inference_steps=2,
        latents_shape=latents_shape,  # 4D or 5D
        device=device,
        dtype=dtype,
    )
    return imgs


def test_sampler_casts_for_vae_decode_image_bf16() -> None:
    imgs = _run(torch.bfloat16, (2, 4, 8, 8))
    assert isinstance(imgs, torch.Tensor)
    assert imgs.ndim == 4


def test_sampler_casts_for_vae_decode_video_bf16() -> None:
    imgs = _run(torch.bfloat16, (1, 4, 2, 8, 8))
    assert isinstance(imgs, torch.Tensor)
    assert imgs.ndim == 5
