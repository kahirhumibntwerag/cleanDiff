from __future__ import annotations

from typing import Iterable

import torch
from torch.optim import Optimizer

from src.domain.interfaces.optimizer import OptimizerBuilder
from src.domain.interfaces.optim_spec import OptimizerKind, OptimizerSpec


class TorchOptimizerBuilder(OptimizerBuilder):
    def build(self, *, spec: OptimizerSpec, params: Iterable[torch.nn.Parameter]) -> Optimizer:
        if spec.kind == OptimizerKind.ADAMW:
            betas = spec.betas if spec.betas is not None else (0.9, 0.999)
            eps = spec.eps if spec.eps is not None else 1e-8
            return torch.optim.AdamW(params, lr=spec.learning_rate, betas=betas, weight_decay=spec.weight_decay, eps=eps)
        if spec.kind == OptimizerKind.ADAM:
            betas = spec.betas if spec.betas is not None else (0.9, 0.999)
            eps = spec.eps if spec.eps is not None else 1e-8
            return torch.optim.Adam(params, lr=spec.learning_rate, betas=betas, weight_decay=spec.weight_decay, eps=eps)
        if spec.kind == OptimizerKind.SGD:
            momentum = spec.momentum if spec.momentum is not None else 0.0
            return torch.optim.SGD(params, lr=spec.learning_rate, weight_decay=spec.weight_decay, momentum=momentum)
        raise ValueError(f"Unsupported optimizer kind: {spec.kind}")


