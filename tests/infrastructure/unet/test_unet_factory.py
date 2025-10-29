from __future__ import annotations

import types
from typing import Any, Dict, Optional, Sequence

import pytest
import torch


class _FakeUNet:
    def __init__(self, **kwargs: Any) -> None:
        # capture kwargs for assertions
        self.kwargs: Dict[str, Any] = dict(kwargs)
        # mimic a minimal config object used by the wrapper
        self.config = types.SimpleNamespace(
            in_channels=int(kwargs.get("in_channels", 0)),
            out_channels=int(kwargs.get("out_channels", 0)),
            cross_attention_dim=kwargs.get("cross_attention_dim", None),
        )

    def to(self, dtype: Optional[torch.dtype] = None):
        self._dtype = dtype
        return self

    def __call__(
        self,
        *,
        sample: torch.Tensor,
        timestep: torch.Tensor | int,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        added_cond_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = False,
    ):
        # Return zeros of the same shape as input sample, wrapped as tuple per return_dict=False
        return (torch.zeros_like(sample),)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *,
        subfolder: Optional[str] = None,
        revision: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        variant: Optional[str] = None,
        use_safetensors: bool = True,
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
    ) -> "_FakeUNet":
        # record the last call for assertions
        cls._last_from_pretrained_args = {
            "pretrained_model_name_or_path": pretrained_model_name_or_path,
            "subfolder": subfolder,
            "revision": revision,
            "torch_dtype": torch_dtype,
            "variant": variant,
            "use_safetensors": use_safetensors,
            "cache_dir": cache_dir,
            "local_files_only": local_files_only,
        }
        # Return a minimally valid model instance
        return cls(
            sample_size=32,
            in_channels=4,
            out_channels=4,
            block_out_channels=(128, 256, 256),
            down_block_types=("DownBlock3D", "DownBlock3D", "DownBlock3D"),
            up_block_types=("UpBlock3D", "UpBlock3D", "UpBlock3D"),
            layers_per_block=2,
            cross_attention_dim=768,
            num_attention_heads=(12, 12, 12),
        )


def _patch_unet_class(monkeypatch: pytest.MonkeyPatch) -> None:
    # Patch the class reference inside the factory module
    monkeypatch.setattr(
        "src.infrastructure.unet.factory.UNetSpatioTemporalConditionModel",
        _FakeUNet,
        raising=False,
    )


def test_create_unet_build_derives_heads_and_omits_norm(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_unet_class(monkeypatch)
    from src.infrastructure.unet.factory import create_diffusers_video_unet

    backbone = create_diffusers_video_unet(sample_size=32)
    # Access fake model
    model = backbone.model  # type: ignore[attr-defined]
    kwargs = getattr(model, "kwargs", {})

    # May pass norm_num_groups depending on diffusers version; if present, it should be a positive int
    if "norm_num_groups" in kwargs:
        assert isinstance(kwargs["norm_num_groups"], int) and kwargs["norm_num_groups"] > 0

    # Should derive num_attention_heads with length equal to number of down blocks
    heads = kwargs.get("num_attention_heads")
    down_blocks: Sequence[str] = kwargs["down_block_types"]
    assert isinstance(heads, (list, tuple)) and len(heads) == len(down_blocks)

    # Expect per-block heads derived from block_out_channels (~64 dims/head)
    assert tuple(heads) == (2, 4, 4)


def test_forward_shapes_for_4d_and_5d(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_unet_class(monkeypatch)
    from src.infrastructure.unet.factory import create_diffusers_video_unet

    backbone = create_diffusers_video_unet(sample_size=32)

    # 4D input should be squeezed back to 4D on output
    latents_4d = torch.randn(2, 4, 32, 32)
    t = torch.tensor([0])
    out_4d = backbone(latents_4d, t)
    assert out_4d.shape == latents_4d.shape

    # 5D input should preserve 5D
    latents_5d = torch.randn(2, 4, 2, 32, 32)
    out_5d = backbone(latents_5d, t)
    assert out_5d.shape == latents_5d.shape


def test_create_unet_from_pretrained_wraps_model(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_unet_class(monkeypatch)
    from src.infrastructure.unet.factory import create_diffusers_video_unet_from_pretrained

    backbone = create_diffusers_video_unet_from_pretrained("dummy/repo", subfolder=None)

    # Returned wrapper should expose a model that is our fake
    assert hasattr(backbone, "model")
    assert isinstance(backbone.model, _FakeUNet)  # type: ignore[attr-defined]

    # Ensure classmethod was invoked
    assert hasattr(_FakeUNet, "_last_from_pretrained_args")
    args = _FakeUNet._last_from_pretrained_args  # type: ignore[attr-defined]
    assert args["pretrained_model_name_or_path"] == "dummy/repo"


def test_real_unet_build_initializes_and_is_correct_type():
    diffusers = pytest.importorskip("diffusers")
    from diffusers import UNetSpatioTemporalConditionModel
    from src.infrastructure.unet.factory import create_diffusers_video_unet

    backbone = create_diffusers_video_unet(sample_size=16, cross_attention_dim=64)
    assert isinstance(backbone.model, UNetSpatioTemporalConditionModel)  # type: ignore[attr-defined]


def test_real_unet_forward_cpu_4d_and_5d():
    pytest.importorskip("diffusers")
    from src.infrastructure.unet.factory import create_diffusers_video_unet

    backbone = create_diffusers_video_unet(sample_size=16, cross_attention_dim=64)

    # Use small shapes to keep tests light; provide encoder_hidden_states to satisfy cross-attention paths
    b, c, f, h, w = 2, 4, 2, 16, 16
    t = torch.tensor([0])
    ehs = torch.randn(b, 1, 64)

    # 4D path
    lat_4d = torch.randn(b, c, h, w)
    out_4d = backbone(lat_4d, t, encoder_hidden_states=ehs)
    assert out_4d.shape == lat_4d.shape

    # 5D path
    lat_5d = torch.randn(b, c, f, h, w)
    out_5d = backbone(lat_5d, t, encoder_hidden_states=ehs)
    assert out_5d.shape == lat_5d.shape


def test_real_unet_head_dim_matches_block_channels():
    pytest.importorskip("diffusers")
    from src.infrastructure.unet.factory import create_diffusers_video_unet

    # Use channels where bad defaults would previously misalign (e.g., 256)
    backbone = create_diffusers_video_unet(sample_size=16, cross_attention_dim=64, block_out_channels=(128, 256, 256))

    cfg = backbone.model.config  # type: ignore[attr-defined]
    # num_attention_heads and attention_head_dim can be int or tuple; coerce to tuple of length blocks
    num_blocks = len(cfg.block_out_channels)
    heads = cfg.num_attention_heads
    # Config may be a FrozenDict; support both attribute and mapping access
    head_dims = getattr(cfg, "attention_head_dim", None)
    if head_dims is None and hasattr(cfg, "__getitem__"):
        try:
            head_dims = cfg["attention_head_dim"]
        except Exception:
            head_dims = None
    if head_dims is None:
        pytest.skip("attention_head_dim not exposed in this diffusers version")
    if isinstance(heads, int):
        heads = (heads,) * num_blocks
    if isinstance(head_dims, int):
        head_dims = (head_dims,) * num_blocks

    for out_ch, h, d in zip(cfg.block_out_channels, heads, head_dims):
        assert out_ch % d == 0
        assert (out_ch // d) == h


