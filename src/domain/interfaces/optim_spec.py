from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


class OptimizerKind(str, Enum):
    ADAMW = "adamw"
    ADAM = "adam"
    SGD = "sgd"


@dataclass
class OptimizerSpec:
    kind: OptimizerKind = OptimizerKind.ADAMW
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    betas: Optional[Tuple[float, float]] = (0.9, 0.999)
    eps: Optional[float] = None
    momentum: Optional[float] = None


