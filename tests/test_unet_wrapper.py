from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from src.infrastructure.unet.diffusers_video_unet import DiffusersVideoUNetBackbone


class _FakeConfig:
    def __init__(self, in_channels: int = 4, out_channels: int = 4, cross_attention_dim: Optional[int] = 16) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cross_attention_dim = cross_attention_dim


class _FakeSpatioTemporalUNet:
    def __init__(self) -> None:
        self.config = _FakeConfig()
        self.last_call: Dict[str, Any] | None = None

    def __call__(
        self,
        *,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        added_cond_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = False,
    ):
        # record inputs for assertions
        self.last_call = {
            "sample_shape": tuple(sample.shape),
            "timestep": timestep,
            "has_ehs": encoder_hidden_states is not None,
            "has_added": added_cond_kwargs is not None,
            "return_dict": return_dict,
        }
        out = torch.zeros_like(sample)
        if return_dict:
            return {"sample": out}
        return (out,)


def test_unet_properties_reflect_config() -> None:
    base = _FakeSpatioTemporalUNet()
    unet = DiffusersVideoUNetBackbone(model=base)
    assert unet.in_channels == base.config.in_channels
    assert unet.out_channels == base.config.out_channels
    assert unet.cross_attention_dim == base.config.cross_attention_dim


def test_forward_with_4d_latents_adds_and_squeezes_frame_dim() -> None:
    base = _FakeSpatioTemporalUNet()
    unet = DiffusersVideoUNetBackbone(model=base)
    x = torch.randn(2, 4, 8, 8)
    t = torch.randint(0, 10, (2,))
    ehs = torch.randn(2, 1, base.config.cross_attention_dim or 16)
    out = unet(x, t, encoder_hidden_states=ehs, added_cond_kwargs={"foo": torch.tensor(1)})
    assert out.shape == x.shape
    # internal call should have had a frames dim of 1
    assert base.last_call is not None
    assert base.last_call["sample_shape"] == (2, 4, 1, 8, 8)
    assert base.last_call["has_ehs"] is True
    assert base.last_call["has_added"] is True


def test_forward_with_5d_latents_preserves_shape() -> None:
    base = _FakeSpatioTemporalUNet()
    unet = DiffusersVideoUNetBackbone(model=base)
    x = torch.randn(2, 4, 3, 8, 8)
    t = torch.randint(0, 10, (2,))
    out = unet(x, t, encoder_hidden_states=None, added_cond_kwargs=None)
    assert out.shape == x.shape
    assert base.last_call is not None
    assert base.last_call["sample_shape"] == tuple(x.shape)


def test_forward_invalid_dims_raises() -> None:
    base = _FakeSpatioTemporalUNet()
    unet = DiffusersVideoUNetBackbone(model=base)
    x = torch.randn(2, 4, 8)  # 3D invalid
    t = torch.randint(0, 10, (2,))
    try:
        _ = unet(x, t)
        assert False, "Expected ValueError for invalid latent dims"
    except ValueError:
        pass


