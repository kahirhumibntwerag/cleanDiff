from __future__ import annotations

import torch

from src.application.usecases.with_encoder_module import TrainEncoderDiffusionModule
from src.domain.interfaces.types import PredictionType

from tests.fakes import (
    FakeEncoder,
    FakeOptimizerBuilder,
    FakeSampler,
    FakeScheduler,
    FakeUNet,
    FakeVAE,
)


def _make_module(pred_type: PredictionType = PredictionType.EPSILON, *, encoder_returns_dict: bool = False) -> TrainEncoderDiffusionModule:
    vae = FakeVAE()
    unet = FakeUNet()
    scheduler = FakeScheduler(prediction_type=pred_type)
    encoder = FakeEncoder(return_dict_even_if_false=encoder_returns_dict)
    opt_builder = FakeOptimizerBuilder()
    sampler = FakeSampler()
    return TrainEncoderDiffusionModule(
        vae=vae,
        unet=unet,
        scheduler=scheduler,
        encoder=encoder,
        optimizer_builder=opt_builder,
        sampler=sampler,
    )


def test_init_freezes_vae_and_trains_encoder() -> None:
    m = _make_module()
    assert all(not p.requires_grad for p in m.vae.parameters())
    assert any(p.requires_grad for p in m.encoder.parameters())


def test_extract_encoder_hidden_states_prefers_existing_in_batch() -> None:
    m = _make_module()
    ehs = torch.randn(2, 4, 16)
    batch = {"pixel_values": torch.randn(2, 3, 64, 64), "encoder_hidden_states": ehs}
    out = m._extract_encoder_hidden_states(batch)
    assert out is ehs


def test_extract_encoder_hidden_states_from_encoder_tensor() -> None:
    m = _make_module(encoder_returns_dict=False)
    enc_inputs = torch.randn(2, 4, 16)
    batch = {"pixel_values": torch.randn(2, 3, 64, 64), "encoder_inputs": enc_inputs}
    out = m._extract_encoder_hidden_states(batch)
    assert isinstance(out, torch.Tensor)


def test_extract_encoder_hidden_states_from_encoder_dict() -> None:
    m = _make_module(encoder_returns_dict=True)
    enc_inputs = {"x": torch.randn(2, 4, 16)}
    batch = {"pixel_values": torch.randn(2, 3, 64, 64), "encoder_inputs": enc_inputs}
    out = m._extract_encoder_hidden_states(batch)
    assert isinstance(out, torch.Tensor)


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


def test_configure_optimizers_returns_two_optimizers() -> None:
    m = _make_module()
    optimizers = m.configure_optimizers()
    assert isinstance(optimizers, list)
    assert len(optimizers) == 2


def test_sample_uses_sampler_with_ehs_or_inputs() -> None:
    m = _make_module()
    # direct ehs
    out1 = m.sample(num_inference_steps=5, latents_shape=(2, m.vae.latent_channels, 8, 8), encoder_hidden_states=torch.randn(2, 4, 16))
    assert isinstance(out1, torch.Tensor)
    # supplied inputs resolve via encoder
    out2 = m.sample(num_inference_steps=5, latents_shape=(2, m.vae.latent_channels, 8, 8), encoder_inputs=torch.randn(2, 4, 16))
    assert isinstance(out2, torch.Tensor)


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


