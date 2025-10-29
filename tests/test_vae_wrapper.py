from __future__ import annotations

from typing import Any

import torch

from src.infrastructure.vae.diffusers_vae import DiffusersVAEBackbone


class _FakeLatentDist:
    def __init__(self, mean: torch.Tensor, logvar: torch.Tensor | None = None) -> None:
        self.mean = mean
        self._logvar = logvar

    @property
    def logvar(self):
        return self._logvar

    @property
    def std(self):
        if self._logvar is None:
            return None
        return (self._logvar.exp()).sqrt()

    @property
    def var(self):
        if self._logvar is None:
            return None
        return self._logvar.exp()

    def sample(self) -> torch.Tensor:
        if self._logvar is None:
            return self.mean
        eps = torch.zeros_like(self.mean)
        return self.mean + (0.5 * self._logvar).exp() * eps


class _FakeEncodeOut:
    def __init__(self, latent_dist: _FakeLatentDist) -> None:
        self.latent_dist = latent_dist


class _FakeDecodeOut:
    def __init__(self, sample: torch.Tensor) -> None:
        self.sample = sample


class _FakeAutoencoder:
    def __init__(self, latent_channels: int = 4, scaling_factor: float = 0.25) -> None:
        class _Cfg:
            pass

        self.config = _Cfg()
        self.config.latent_channels = latent_channels
        self.config.scaling_factor = scaling_factor
        self.last_decode_in: torch.Tensor | None = None

    def encode(self, images: torch.Tensor) -> _FakeEncodeOut:
        B, _, H, W = images.shape
        latents = torch.zeros(B, self.config.latent_channels, H // 8, W // 8, device=images.device, dtype=images.dtype)
        mean = torch.zeros_like(latents)
        logvar = torch.zeros_like(latents)
        return _FakeEncodeOut(_FakeLatentDist(mean=mean, logvar=logvar))

    def decode(self, latents: torch.Tensor) -> _FakeDecodeOut:
        # record input for assertions
        self.last_decode_in = latents
        B, C, H, W = latents.shape
        img = torch.zeros(B, 3, H * 8, W * 8, device=latents.device, dtype=latents.dtype)
        return _FakeDecodeOut(img)


def test_properties_exposed_from_config() -> None:
    model = _FakeAutoencoder(latent_channels=8, scaling_factor=0.5)
    vae = DiffusersVAEBackbone(model=model)
    assert vae.latent_channels == 8
    assert vae.scaling_factor == 0.5


def test_encode_returns_dataclass_fields() -> None:
    model = _FakeAutoencoder()
    vae = DiffusersVAEBackbone(model=model)
    images = torch.randn(2, 3, 64, 64)
    out = vae.encode(images, sample=True)
    assert out.latents.shape == (2, model.config.latent_channels, 8, 8)
    assert out.mean.shape == out.latents.shape
    assert out.logvar.shape == out.latents.shape


def test_decode_divides_by_scaling_factor() -> None:
    model = _FakeAutoencoder(latent_channels=4, scaling_factor=0.5)
    vae = DiffusersVAEBackbone(model=model)
    latents = torch.ones(2, 4, 8, 8)
    _ = vae.decode(latents)
    assert model.last_decode_in is not None
    # last_decode_in should be latents / scaling_factor
    assert torch.allclose(model.last_decode_in, latents / model.config.scaling_factor)


