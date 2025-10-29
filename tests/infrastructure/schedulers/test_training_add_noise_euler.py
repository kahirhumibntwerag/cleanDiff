from __future__ import annotations

import pytest
import torch

from src.application.usecases.without_encoder_module import TrainWithoutEncoderDiffusionModule
from src.application.usecases.with_encoder_module import TrainEncoderDiffusionModule
from src.infrastructure.schedulers.factory import create_diffusers_scheduler
from tests.fakes import FakeOptimizerBuilder, FakeSampler, FakeUNet, FakeVAE, FakeEncoder


pytestmark = pytest.mark.skipif(pytest.importorskip("diffusers", reason="diffusers not installed") is None, reason="diffusers not installed")  # type: ignore[truthy-bool]


def _batch(image_size: int = 16) -> dict[str, torch.Tensor]:
    return {"pixel_values": torch.randn(2, 3, image_size, image_size)}


def test_training_step_initializes_euler_timesteps_without_encoder():
    vae = FakeVAE(latent_channels=4, scaling_factor=0.5)
    unet = FakeUNet(in_channels=4, out_channels=4)
    scheduler = create_diffusers_scheduler(kind="euler")
    sampler = FakeSampler()
    opt_builder = FakeOptimizerBuilder()

    module = TrainWithoutEncoderDiffusionModule(vae=vae, unet=unet, scheduler=scheduler, optimizer_builder=opt_builder, sampler=sampler)
    loss = module.training_step(_batch(), 0)
    assert isinstance(loss, torch.Tensor)


def test_training_step_initializes_euler_timesteps_with_encoder():
    vae = FakeVAE(latent_channels=4, scaling_factor=0.5)
    unet = FakeUNet(in_channels=4, out_channels=4)
    scheduler = create_diffusers_scheduler(kind="euler")
    sampler = FakeSampler()
    opt_builder = FakeOptimizerBuilder()
    enc = FakeEncoder()

    module = TrainEncoderDiffusionModule(vae=vae, unet=unet, scheduler=scheduler, encoder=enc, optimizer_builder=opt_builder, sampler=sampler)
    loss = module.training_step(_batch(), 0)
    assert isinstance(loss, torch.Tensor)
