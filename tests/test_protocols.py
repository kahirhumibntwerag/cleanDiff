from __future__ import annotations

from typing import Dict, Iterable, Sequence

import torch

from src.domain.interfaces.dataset import DiffusionDataset
from src.domain.interfaces.datamodule import DiffusionDataModule
from src.domain.interfaces.encoder import Encoder
from src.domain.interfaces.optimizer import OptimizerBuilder
from src.domain.interfaces.optim_spec import OptimizerSpec
from src.domain.interfaces.sampler import Sampler
from src.domain.interfaces.scheduler import Scheduler
from src.domain.interfaces.types import DiffusionBatch, PredictionType, SchedulerStepOutput
from src.domain.interfaces.unet import UNetBackbone
from src.domain.interfaces.vae import VAE


class _DatasetImpl:
    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> DiffusionBatch:
        return {"pixel_values": torch.randn(1, 3, 8, 8)}

    def collate_fn(self, batch: Sequence[DiffusionBatch]) -> DiffusionBatch:
        return batch[0]


class _DataModuleImpl:
    batch_size = 2

    def setup(self, stage: str | None = None) -> None:
        return None

    def train_dataloader(self):  # type: ignore[override]
        class _DL:
            def __iter__(self):
                yield {"pixel_values": torch.randn(2, 3, 8, 8)}

        return _DL()

    val_dataloader = train_dataloader
    test_dataloader = train_dataloader
    predict_dataloader = train_dataloader


class _EncoderImpl:
    @property
    def hidden_size(self) -> int:
        return 16

    def forward(self, inputs: torch.Tensor | Dict[str, torch.Tensor], *, return_dict: bool = False):
        hs = torch.randn(2, 4, self.hidden_size)
        return {"encoder_hidden_states": hs} if return_dict else hs

    __call__ = forward

    def to(self, device: torch.device, dtype: torch.dtype | None = None):
        return self

    def parameters(self) -> Iterable[torch.nn.Parameter]:
        return []

    def optimizer_spec(self) -> OptimizerSpec:
        return OptimizerSpec()


class _UNetImpl:
    @property
    def in_channels(self) -> int:
        return 4

    @property
    def out_channels(self) -> int:
        return 4

    def forward(self, latents: torch.Tensor, timesteps: torch.Tensor, encoder_hidden_states: torch.Tensor | None = None, added_cond_kwargs: dict | None = None) -> torch.Tensor:  # noqa: E501
        return torch.zeros_like(latents)

    __call__ = forward

    def parameters(self) -> Iterable[torch.nn.Parameter]:
        return []

    def optimizer_spec(self) -> OptimizerSpec:
        return OptimizerSpec()


class _VAEImpl:
    @property
    def latent_channels(self) -> int:
        return 4

    @property
    def scaling_factor(self) -> float:
        return 0.5

    def encode(self, images: torch.Tensor, sample: bool = True):
        from src.domain.interfaces.types import VAEEncodeOutput

        latents = torch.randn(images.size(0), 4, images.size(2) // 8, images.size(3) // 8)
        return VAEEncodeOutput(latents=latents, mean=torch.zeros_like(latents), logvar=torch.zeros_like(latents))

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return torch.randn(latents.size(0), 3, latents.size(2) * 8, latents.size(3) * 8)

    def to(self, device: torch.device, dtype: torch.dtype | None = None):
        return self

    def parameters(self) -> Iterable[torch.nn.Parameter]:
        return []

    def optimizer_spec(self) -> OptimizerSpec:
        return OptimizerSpec()


class _SchedulerImpl:
    @property
    def num_train_timesteps(self) -> int:
        return 1000

    @property
    def init_noise_sigma(self) -> float:
        return 1.0

    @property
    def prediction_type(self) -> PredictionType:
        return PredictionType.EPSILON

    def set_timesteps(self, num_inference_steps: int, device: torch.device) -> torch.Tensor:
        return torch.arange(num_inference_steps, device=device)

    def scale_model_input(self, sample: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        return sample

    def step(self, model_output: torch.Tensor, timestep: torch.Tensor, sample: torch.Tensor, return_dict: bool = True) -> SchedulerStepOutput:  # noqa: E501
        return SchedulerStepOutput(prev_sample=sample)

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return original_samples + noise

    def get_velocity(self, sample: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        return noise


class _SamplerImpl:
    @torch.no_grad()
    def sample(self, *, unet: UNetBackbone, scheduler: Scheduler, vae: VAE, encoder_hidden_states: torch.Tensor | None, num_inference_steps: int, latents_shape: tuple[int, int, int, int], device: torch.device | None = None, dtype: torch.dtype | None = None) -> torch.Tensor:  # noqa: E501
        return torch.zeros(latents_shape, device=device, dtype=dtype)


class _OptBuilderImpl:
    def build(self, *, spec: OptimizerSpec, params: Iterable[torch.nn.Parameter]):
        return torch.optim.SGD(params, lr=spec.learning_rate)


def test_dataset_protocol_isinstance() -> None:
    assert isinstance(_DatasetImpl(), DiffusionDataset)


def test_datamodule_protocol_isinstance() -> None:
    assert isinstance(_DataModuleImpl(), DiffusionDataModule)


def test_encoder_protocol_isinstance() -> None:
    assert isinstance(_EncoderImpl(), Encoder)


def test_unet_protocol_isinstance() -> None:
    assert isinstance(_UNetImpl(), UNetBackbone)


def test_vae_protocol_isinstance() -> None:
    assert isinstance(_VAEImpl(), VAE)


def test_scheduler_protocol_isinstance() -> None:
    assert isinstance(_SchedulerImpl(), Scheduler)


def test_sampler_protocol_isinstance() -> None:
    assert isinstance(_SamplerImpl(), Sampler)


def test_optimizer_builder_protocol_isinstance() -> None:
    assert isinstance(_OptBuilderImpl(), OptimizerBuilder)


def test_protocol_negative_missing_method_fails_isinstance() -> None:
    class _BadUNet:
        def forward(self, latents: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:  # missing other args and properties
            return latents

        __call__ = forward

    assert not isinstance(_BadUNet(), UNetBackbone)


