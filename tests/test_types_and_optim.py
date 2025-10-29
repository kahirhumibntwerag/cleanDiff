from __future__ import annotations

import torch

from src.domain.interfaces.types import (
    PredictionType,
    VAEEncodeOutput,
    SchedulerStepOutput,
)
from src.domain.interfaces.optim_spec import OptimizerKind, OptimizerSpec
from src.infrastructure.optimizers.torch_builder import TorchOptimizerBuilder


def test_prediction_type_values() -> None:
    assert PredictionType.EPSILON.value == "epsilon"
    assert PredictionType.V_PREDICTION.value == "v_prediction"
    assert PredictionType.SAMPLE.value == "sample"


def test_vae_encode_output_dataclass_fields() -> None:
    latents = torch.randn(2, 4, 8, 8)
    mean = torch.zeros_like(latents)
    logvar = torch.ones_like(latents)
    out = VAEEncodeOutput(latents=latents, mean=mean, logvar=logvar)
    assert out.latents is latents
    assert out.mean is mean
    assert out.logvar is logvar


def test_scheduler_step_output_dataclass_fields() -> None:
    prev = torch.randn(2, 4, 8, 8)
    x0 = torch.randn(2, 4, 8, 8)
    step = SchedulerStepOutput(prev_sample=prev, pred_original_sample=x0)
    assert step.prev_sample is prev
    assert step.pred_original_sample is x0


def test_optimizer_spec_defaults_and_customization() -> None:
    spec = OptimizerSpec()
    assert spec.kind == OptimizerKind.ADAMW
    assert spec.learning_rate == 1e-4
    assert spec.weight_decay == 1e-2
    assert spec.betas == (0.9, 0.999)

    custom = OptimizerSpec(
        kind=OptimizerKind.SGD,
        learning_rate=5e-3,
        weight_decay=0.0,
        momentum=0.9,
    )
    assert custom.kind == OptimizerKind.SGD
    assert custom.learning_rate == 5e-3
    assert custom.weight_decay == 0.0
    assert custom.momentum == 0.9


def test_torch_optimizer_builder_builds_expected_optimizers() -> None:
    builder = TorchOptimizerBuilder()
    params = list(torch.nn.Linear(2, 2).parameters())

    adamw = builder.build(spec=OptimizerSpec(kind=OptimizerKind.ADAMW), params=params)
    assert isinstance(adamw, torch.optim.AdamW)

    adam = builder.build(spec=OptimizerSpec(kind=OptimizerKind.ADAM), params=params)
    assert isinstance(adam, torch.optim.Adam)

    sgd = builder.build(spec=OptimizerSpec(kind=OptimizerKind.SGD), params=params)
    assert isinstance(sgd, torch.optim.SGD)


