from __future__ import annotations

import torch

from src.application.usecases.without_encoder_module import TrainWithoutEncoderDiffusionModule
from src.domain.interfaces.types import PredictionType

from tests.fakes import FakeOptimizerBuilder, FakeSampler, FakeScheduler, FakeUNet, FakeVAE


def _make_module(pred_type: PredictionType = PredictionType.EPSILON) -> TrainWithoutEncoderDiffusionModule:
    vae = FakeVAE()
    unet = FakeUNet()
    scheduler = FakeScheduler(prediction_type=pred_type)
    opt_builder = FakeOptimizerBuilder()
    sampler = FakeSampler()
    return TrainWithoutEncoderDiffusionModule(
        vae=vae,
        unet=unet,
        scheduler=scheduler,
        optimizer_builder=opt_builder,
        sampler=sampler,
    )


def test_init_freezes_vae_requires_grad() -> None:
    m = _make_module()
    assert all(not p.requires_grad for p in m.vae.parameters())


def test_training_step_runs_and_returns_loss_scalar() -> None:
    m = _make_module(PredictionType.EPSILON)
    batch = {"pixel_values": torch.randn(2, 3, 64, 64)}
    loss = m.training_step(batch, 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0


def test_validation_step_runs() -> None:
    m = _make_module(PredictionType.EPSILON)
    batch = {"pixel_values": torch.randn(2, 3, 64, 64)}
    loss = m.validation_step(batch, 0)
    assert isinstance(loss, torch.Tensor)


def test_configure_optimizers_returns_optimizer() -> None:
    m = _make_module()
    opt = m.configure_optimizers()
    import torch as _t

    assert isinstance(opt, _t.optim.Optimizer)


def test_sample_uses_sampler_and_returns_latents_shape() -> None:
    m = _make_module()
    out = m.sample(num_inference_steps=5, latents_shape=(2, m.vae.latent_channels, 8, 8))
    assert isinstance(out, torch.Tensor)
    assert tuple(out.shape) == (2, m.vae.latent_channels, 8, 8)


def test_compute_target_for_all_prediction_types() -> None:
    latents = torch.randn(2, 4, 8, 8)
    noise = torch.randn_like(latents)
    t = torch.randint(0, 10, (2,))

    m_eps = _make_module(PredictionType.EPSILON)
    assert torch.allclose(m_eps._compute_target(latents, noise, t), noise)

    m_v = _make_module(PredictionType.V_PREDICTION)
    assert torch.allclose(m_v._compute_target(latents, noise, t), noise)

    m_x0 = _make_module(PredictionType.SAMPLE)
    assert torch.allclose(m_x0._compute_target(latents, noise, t), latents)


def test_prepare_latents_scales_by_vae_scaling_factor() -> None:
    m = _make_module()
    images = torch.randn(2, 3, 64, 64)
    latents = m._prepare_latents(images, sample_from_posterior=True)
    # FakeVAE.encode returns ones; scaled by scaling_factor
    assert torch.allclose(latents, torch.ones_like(latents) * m.vae.scaling_factor)


