from __future__ import annotations

from typing import Iterable, Protocol, runtime_checkable

import torch
from torch.optim import Optimizer

from .optim_spec import OptimizerSpec


@runtime_checkable
class OptimizerBuilder(Protocol):
    """
    Build a torch optimizer from a spec and parameter iterable.
    """

    def build(self, *, spec: OptimizerSpec, params: Iterable[torch.nn.Parameter]) -> Optimizer:
        ...


