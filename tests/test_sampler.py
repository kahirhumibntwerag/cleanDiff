from __future__ import annotations

import torch

from src.infrastructure.samplers import DiffusersDenoiserSampler

from tests.fakes import FakeScheduler, FakeUNet, FakeVAE


def test_sampler_generates_images_without_conditioning() -> None:
    unet = FakeUNet(in_channels=4, out_channels=4)
    scheduler = FakeScheduler()
    vae = FakeVAE(latent_channels=4, scaling_factor=0.5)
    sampler = DiffusersDenoiserSampler()

    images = sampler.sample(
        unet=unet,
        scheduler=scheduler,
        vae=vae,
        encoder_hidden_states=None,
        num_inference_steps=3,
        latents_shape=(2, vae.latent_channels, 8, 8),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert isinstance(images, torch.Tensor)
    assert tuple(images.shape) == (2, 3, 64, 64)


def test_sampler_generates_images_with_conditioning() -> None:
    unet = FakeUNet(in_channels=4, out_channels=4)
    scheduler = FakeScheduler()
    vae = FakeVAE(latent_channels=4, scaling_factor=0.5)
    sampler = DiffusersDenoiserSampler()
    ehs = torch.zeros(2, 1, 16)

    images = sampler.sample(
        unet=unet,
        scheduler=scheduler,
        vae=vae,
        encoder_hidden_states=ehs,
        num_inference_steps=3,
        latents_shape=(2, vae.latent_channels, 8, 8),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert isinstance(images, torch.Tensor)
    assert tuple(images.shape) == (2, 3, 64, 64)


