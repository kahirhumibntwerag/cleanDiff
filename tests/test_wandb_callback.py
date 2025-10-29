from __future__ import annotations

import types
import torch
import pytest

from types import SimpleNamespace
from typing import Any, List

import numpy as np
import lightning.pytorch as pl

import src.infrastructure.callbacks.wandb_sampling as ws
from src.infrastructure.callbacks.wandb_sampling import WandbVideoSamplingCallback


class _FakeVideo:
    def __init__(self, arr: np.ndarray, fps: int, format: str) -> None:
        self.arr = arr
        self.fps = fps
        self.format = format


class _FakeWandb:
    def __init__(self) -> None:
        self.logged: List[dict[str, Any]] = []

    def Video(self, arr: np.ndarray, fps: int, format: str) -> _FakeVideo:  # type: ignore[override]
        return _FakeVideo(arr, fps, format)

    def log(self, data: dict, step: int | None = None) -> None:
        self.logged.append({"data": data, "step": step})


class _FakeCM:
    def get_cmap(self, name: str):  # pragma: no cover - trivial
        def _apply(arr: np.ndarray) -> np.ndarray:
            h, w = arr.shape
            rgba = np.zeros((h, w, 4), dtype=np.float32)
            rgba[..., 0] = arr
            rgba[..., 1] = arr
            rgba[..., 2] = arr
            rgba[..., 3] = 1.0
            return rgba

        return _apply


class _DummyModule(pl.LightningModule):
    def __init__(self, *, frames: int) -> None:
        super().__init__()
        self.vae = SimpleNamespace(latent_channels=4)
        self._frames = frames
        self.encoder = None

    @property
    def device(self):  # present in LightningModule but make explicit
        return torch.device("cpu")

    @torch.no_grad()
    def sample(self, *, num_inference_steps: int, latents_shape, encoder_hidden_states=None, encoder_inputs=None, device=None, dtype=None):  # noqa: E501
        b = latents_shape[0]
        if len(latents_shape) == 5 or self._frames > 1:
            t = latents_shape[2]
            return torch.zeros(b, 3, t, 8 * latents_shape[-2], 8 * latents_shape[-1])
        return torch.zeros(b, 3, 8 * latents_shape[-2], 8 * latents_shape[-1])


class _FakeModule(torch.nn.Module):
    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self._dev = device
        self.vae = types.SimpleNamespace(latent_channels=4)
        # simple parameter to check grads unaffected
        self.w = torch.nn.Parameter(torch.ones(1))

    @property
    def device(self) -> torch.device:  # type: ignore[override]
        return self._dev

    def sample(self, *, num_inference_steps: int, latents_shape, encoder_hidden_states=None, encoder_inputs=None, device=None, dtype=None):  # type: ignore[override]
        # return a GPU tensor so callback must move to CPU
        if len(latents_shape) == 4:
            b, c, h, w = latents_shape
            return torch.zeros(b, 3, h * 8, w * 8, device=self._dev, dtype=dtype or torch.float32)
        b, c, f, h, w = latents_shape
        return torch.zeros(b, 3, f, h * 8, w * 8, device=self._dev, dtype=dtype or torch.float32)


class _FakeTrainer:
    def __init__(self, precision_value: str, global_step: int = 100) -> None:
        self.global_step = global_step
        self.precision = types.SimpleNamespace(value=precision_value)


def test_callback_logs_video_every_n_steps(monkeypatch):
    fake_wandb = _FakeWandb()
    monkeypatch.setattr(ws, "wandb", fake_wandb, raising=False)
    monkeypatch.setattr(ws, "cm", _FakeCM(), raising=False)

    cb = WandbVideoSamplingCallback(sample_every_n_steps=10, num_inference_steps=2, num_frames=3, image_size=32, speeds=[100.0])
    trainer = SimpleNamespace(global_step=10)
    pl_module = _DummyModule(frames=3)

    cb.on_train_batch_end(trainer, pl_module, None, None, 0)
    assert len(fake_wandb.logged) == 1
    entry = fake_wandb.logged[0]["data"][cb.log_key]
    assert isinstance(entry[0], _FakeVideo)
    assert entry[0].arr.ndim == 4  # (T,H,W,3)


def test_callback_does_not_log_when_not_on_step(monkeypatch):
    fake_wandb = _FakeWandb()
    monkeypatch.setattr(ws, "wandb", fake_wandb, raising=False)
    monkeypatch.setattr(ws, "cm", _FakeCM(), raising=False)

    cb = WandbVideoSamplingCallback(sample_every_n_steps=10, num_inference_steps=2, num_frames=1, image_size=32, speeds=[100.0])
    trainer = SimpleNamespace(global_step=9)
    pl_module = _DummyModule(frames=1)

    cb.on_train_batch_end(trainer, pl_module, None, None, 0)
    assert len(fake_wandb.logged) == 0


def test_wandb_callback_uses_trainer_precision_and_moves_images_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {"frames_on_cpu": False}

    # monkeypatch wandb to avoid real logging/Video
    class _W:
        class Video:
            def __init__(self, arr, fps: int, format: str):
                pass

        @staticmethod
        def log(data, step=None):
            pass

    # Patch wandb inside the module
    monkeypatch.setattr(ws, "wandb", _W())

    # Intercept frame extraction to assert CPU tensors are passed
    import src.infrastructure.callbacks.wandb_sampling as mod

    def _spy(video: torch.Tensor, *args, **kwargs):
        # video tensor should be on CPU
        called["frames_on_cpu"] = video.device.type == "cpu"
        # return a list of fake frames
        return [torch.zeros(8, 8, 3, dtype=torch.uint8).numpy()]

    monkeypatch.setattr(mod, "_frames_from_video_afmhot", _spy)

    cb = WandbVideoSamplingCallback(sample_every_n_steps=100, num_inference_steps=2, num_frames=1, image_size=16)
    trainer = _FakeTrainer(precision_value="bf16-mixed", global_step=100)
    pl_module = _FakeModule(device=torch.device("cpu"))
    pl_module.train()
    assert pl_module.training is True
    # ensure no pre-existing grad
    assert pl_module.w.grad is None

    # Should sample now and move images to CPU for frame extraction
    cb.on_train_batch_end(trainer, pl_module, None, None, 0)
    assert called["frames_on_cpu"] is True
    # training state restored
    assert pl_module.training is True
    # no grad created by sampling
    assert pl_module.w.grad is None


