from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Union, TypedDict

import torch


class PredictionType(str, Enum):
    EPSILON = "epsilon"          # predict noise ε
    V_PREDICTION = "v_prediction"  # predict velocity v
    SAMPLE = "sample"            # predict x0


@dataclass
class VAEEncodeOutput:
    latents: torch.Tensor              # [B, C_latent, H_latent, W_latent]
    mean: torch.Tensor                 # [B, C_latent, H_latent, W_latent]
    logvar: torch.Tensor               # [B, C_latent, H_latent, W_latent]


@dataclass
class SchedulerStepOutput:
    prev_sample: torch.Tensor                  # x_{t-1}
    pred_original_sample: Optional[torch.Tensor] = None  # optional x0 estimate


# Encoder input contract and batch typing
EncoderInputs = Union[torch.Tensor, Dict[str, torch.Tensor]]


class DiffusionBatchBase(TypedDict):
    """
    Base batch fields required by the diffusion training loop.
    """
    pixel_values: torch.Tensor  # [B, 3, H, W], float, typically in [-1, 1]


class DiffusionBatch(DiffusionBatchBase, total=False):
    """
    Optional fields that can be provided by datasets or datamodules.
    If the UNet is conditioned, the batch MUST include exactly one of:
      - `encoder_hidden_states` (precomputed), or
      - `encoder_inputs` (raw inputs for the Encoder)
    """

    # Precomputed conditioning to pass directly to the UNet
    encoder_hidden_states: torch.Tensor  # [B, S, D] or [B, D]

    # Inputs to an Encoder implementation
    encoder_inputs: EncoderInputs



