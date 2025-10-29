from __future__ import annotations

import types
from typing import Any

import pytest


class _RejectAttentionHeadDimUNet:
    """Fake UNet that rejects attention_head_dim at init to simulate older diffusers."""

    def __init__(self, **kwargs: Any) -> None:
        if "attention_head_dim" in kwargs:
            raise TypeError("unexpected keyword argument 'attention_head_dim'")
        # capture important config
        self.config = types.SimpleNamespace(
            block_out_channels=tuple(kwargs.get("block_out_channels", ())),
            num_attention_heads=tuple(kwargs.get("num_attention_heads", ())),
        )


def test_factory_fallback_adjusts_block_channels_when_attention_head_dim_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    # Patch the class used by the factory
    monkeypatch.setattr(
        "src.infrastructure.unet.factory.UNetSpatioTemporalConditionModel",
        _RejectAttentionHeadDimUNet,
        raising=False,
    )

    from src.infrastructure.unet.factory import create_diffusers_video_unet

    backbone = create_diffusers_video_unet(sample_size=16, block_out_channels=(128, 256, 256))
    cfg = backbone.model.config  # type: ignore[attr-defined]

    # Expect fallback to multiples of 88 with heads rounded up to a multiple of 4:
    # heads ~ [2,4,4] -> adjusted to [4,4,4] -> block_out_channels [352,352,352]
    assert cfg.block_out_channels == (352, 352, 352)

