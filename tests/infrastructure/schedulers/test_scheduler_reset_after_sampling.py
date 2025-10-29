from __future__ import annotations

import torch

from src.application.usecases.without_encoder_module import TrainWithoutEncoderDiffusionModule
from src.application.usecases.with_encoder_module import TrainEncoderDiffusionModule
from tests.fakes import FakeUNet, FakeVAE, FakeOptimizerBuilder


class _RecordingScheduler:
    def __init__(self, train_steps: int = 1000) -> None:
        self._train = train_steps
        self.last_set = None
        from src.domain.interfaces.types import PredictionType
        self._pred = PredictionType.EPSILON

    @property
    def num_train_timesteps(self) -> int:
        return self._train

    @property
    def prediction_type(self):
        return self._pred

    def set_timesteps(self, num_inference_steps: int, device: torch.device):
        self.last_set = int(num_inference_steps)
        return torch.arange(num_inference_steps, device=device)

    def scale_model_input(self, sample: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        return sample

    def step(self, model_output: torch.Tensor, timestep: torch.Tensor, sample: torch.Tensor, return_dict: bool = True):
        class O:
            def __init__(self, x):
                self.prev_sample = x
        return O(sample)

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return original_samples + noise

    def get_velocity(self, sample: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return noise


class _SamplerThatSetsInference:
    @torch.no_grad()
    def sample(self, *, unet, scheduler, vae, encoder_hidden_states, num_inference_steps, latents_shape, device, dtype):
        # simulate inference schedule being set (e.g., 25)
        scheduler.set_timesteps(num_inference_steps=25, device=device)
        return torch.zeros((latents_shape[0], 3, 8, 8), device=device, dtype=dtype or torch.float32) if len(latents_shape) == 4 else torch.zeros((latents_shape[0], 3, latents_shape[2], 8, 8), device=device, dtype=dtype or torch.float32)


def _batch(image_size: int = 16) -> dict[str, torch.Tensor]:
    return {"pixel_values": torch.randn(2, 3, image_size, image_size)}


def test_scheduler_timesteps_reset_after_sampling_without_encoder() -> None:
    sched = _RecordingScheduler(train_steps=1000)
    module = TrainWithoutEncoderDiffusionModule(
        vae=FakeVAE(latent_channels=4, scaling_factor=0.5),
        unet=FakeUNet(in_channels=4, out_channels=4),
        scheduler=sched,
        optimizer_builder=FakeOptimizerBuilder(),
        sampler=_SamplerThatSetsInference(),
    )
    # initial training step should set training timesteps
    module.training_step(_batch(), 0)
    assert sched.last_set == 1000
    # sampling changes schedule
    module.sample(num_inference_steps=5, latents_shape=(2, 4, 8, 8))
    assert sched.last_set == 25
    # next training step resets schedule to training
    module.training_step(_batch(), 1)
    assert sched.last_set == 1000


def test_scheduler_timesteps_reset_after_sampling_with_encoder() -> None:
    sched = _RecordingScheduler(train_steps=1000)
    from tests.fakes import FakeEncoder, FakeSampler

    module = TrainEncoderDiffusionModule(
        vae=FakeVAE(latent_channels=4, scaling_factor=0.5),
        unet=FakeUNet(in_channels=4, out_channels=4),
        scheduler=sched,
        encoder=FakeEncoder(),
        optimizer_builder=FakeOptimizerBuilder(),
        sampler=_SamplerThatSetsInference(),
    )
    module.training_step(_batch(), 0)
    assert sched.last_set == 1000
    module.sample(num_inference_steps=5, latents_shape=(2, 4, 8, 8), encoder_hidden_states=None)
    assert sched.last_set == 25
    module.training_step(_batch(), 1)
    assert sched.last_set == 1000
