from __future__ import annotations

from typing import List, Optional, Sequence

import math
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch import LightningModule, Trainer

try:
    import wandb  # type: ignore
except Exception:  # pragma: no cover - optional in tests
    wandb = None  # type: ignore

try:
    from matplotlib import cm  # type: ignore
except Exception:  # pragma: no cover - optional in tests
    cm = None  # type: ignore


def _to_linear_pixel_space(
    t: torch.Tensor,
    *,
    log_eps: float = 1e-5,
    max_phys: float = 20000.0,
) -> torch.Tensor:
    if t.ndim not in (3, 4):
        raise ValueError(f"Expected 3D or 4D tensor, got {t.shape}")
    x = t.detach().float()
    x_255 = (x + 1.0) / 2.0 * 255.0
    log_lo = float(math.log10(log_eps))
    log_hi = float(math.log10(max_phys))
    x_log = log_lo + (x_255 / 255.0) * (log_hi - log_lo)
    x_lin = (10.0 ** x_log) - log_eps
    x_lin = torch.clamp(x_lin, min=0.0).to(torch.float32)
    return x_lin


def _frames_from_video_afmhot(
    video: torch.Tensor,  # [B,C,T,H,W] or [C,T,H,W] (B=1)
    *,
    log_eps: float = 1e-5,
    max_phys: float = 20000.0,
) -> List[np.ndarray]:
    if cm is None:
        # fallback to grayscale if matplotlib isn't present
        frames = []
        if video.ndim == 4:
            _, T, H, W = video.shape
            for t in range(T):
                frame = video[:, t, :, :].mean(dim=0).clamp(-1, 1)
                arr = (((frame + 1.0) / 2.0) * 255.0).cpu().numpy().astype(np.uint8)
                rgb = np.stack([arr, arr, arr], axis=-1)
                frames.append(rgb)
            return frames
        _, _, T, H, W = video.shape
        for t in range(T):
            frame = video[0, :, t, :, :].mean(dim=0).clamp(-1, 1)
            arr = (((frame + 1.0) / 2.0) * 255.0).cpu().numpy().astype(np.uint8)
            rgb = np.stack([arr, arr, arr], axis=-1)
            frames.append(rgb)
        return frames

    if video.ndim == 4:
        C, T, H, W = video.shape
        seq = video
    else:
        assert video.shape[0] == 1, "Expected batch size 1 for visualization"
        _, C, T, H, W = video.shape
        seq = video[0]

    frames: List[np.ndarray] = []
    for i in range(T):
        frame_chw = seq[:, i, :, :]  # [C,H,W]
        frame_lin = _to_linear_pixel_space(frame_chw, log_eps=log_eps, max_phys=max_phys)  # [C,H,W]
        frame_gray = frame_lin.mean(dim=0)  # [H,W]
        fg = frame_gray.flatten()
        lo = torch.quantile(fg, 0.01).item()
        hi = torch.quantile(fg, 0.99).item()
        if hi <= lo:
            lo, hi = float(frame_gray.min().item()), float(frame_gray.max().item())
        arr = frame_gray.clamp(min=lo, max=hi)
        if hi > lo:
            arr = (arr - lo) / (hi - lo)
        else:
            # Uniform frame; render mid-gray to avoid fully black output
            arr = torch.full_like(arr, 0.5)
        arr_np = arr.cpu().numpy()
        rgba = cm.get_cmap("afmhot")(arr_np)
        rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
        frames.append(rgb)
    return frames


class WandbVideoSamplingCallback(Callback):
    def __init__(
        self,
        *,
        sample_every_n_steps: int = 200,
        num_inference_steps: int = 25,
        num_frames: int = 8,
        image_size: int = 256,
        speeds: Optional[Sequence[float]] = None,
        log_key: str = "samples/video",
        fps: int = 4,
    ) -> None:
        super().__init__()
        self.sample_every_n_steps = int(sample_every_n_steps)
        self.num_inference_steps = int(num_inference_steps)
        self.num_frames = int(num_frames)
        self.image_size = int(image_size)
        self.speeds = list(speeds) if speeds is not None else [100.0, 200.0, 300.0]
        self.log_key = log_key
        self.fps = int(fps)

    def _maybe_sample_and_log(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if wandb is None:
            return
        if not trainer.global_step or trainer.global_step % self.sample_every_n_steps != 0:
            return

        device = pl_module.device
        # Choose dtype consistent with Trainer precision, but avoid float16 on CPU
        prec = getattr(getattr(trainer, "precision", None), "value", None) or getattr(trainer, "precision", None)
        dev_type = pl_module.device.type if hasattr(pl_module, "device") else ("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(prec, str):
            low = prec.lower()
            if "bf16" in low:
                dtype = torch.bfloat16
            elif "16" in low:
                # Prefer bf16 on CPU; fp16 on CUDA
                dtype = torch.float16 if dev_type == "cuda" else torch.bfloat16
            else:
                dtype = torch.float32
        else:
            # Default: mixed precision only on CUDA; otherwise use float32 on CPU
            if torch.cuda.is_available() and dev_type == "cuda":
                dtype = torch.float16
            else:
                dtype = torch.float32

        # Decide if model supports video sampling: if UNet expects 5D and sampler/vae support, sample video
        # We'll always construct video latents [B,C,F,H,W] if num_frames > 1
        b = len(self.speeds)
        c = getattr(pl_module.vae, "latent_channels", 4)
        h = self.image_size // 8
        w = self.image_size // 8

        # Prepare conditioning
        encoder_hidden_states = None
        encoder_inputs = None
        if hasattr(pl_module, "encoder") and pl_module.encoder is not None:
            # Build encoder inputs tensor [B,1]
            encoder_inputs = torch.tensor(self.speeds, dtype=dtype, device=device)[:, None]

        # Call sample on module; it returns images
        if hasattr(pl_module, "sample"):
            # Temporarily set eval to minimize training-mode buffers during sampling
            was_training = bool(pl_module.training)
            try:
                pl_module.eval()
                # Ensure no autograd graph and minimal allocation
                with torch.inference_mode():
                    if encoder_inputs is not None:
                        images = pl_module.sample(
                            num_inference_steps=self.num_inference_steps,
                            latents_shape=(b, c, self.num_frames, h, w) if self.num_frames > 1 else (b, c, h, w),
                            encoder_inputs=encoder_inputs,  # type: ignore[arg-type]
                            device=device,
                            dtype=dtype,
                        )
                    else:
                        images = pl_module.sample(
                            num_inference_steps=self.num_inference_steps,
                            latents_shape=(b, c, self.num_frames, h, w) if self.num_frames > 1 else (b, c, h, w),
                            encoder_hidden_states=encoder_hidden_states,
                            device=device,
                            dtype=dtype,
                        )
            finally:
                if was_training:
                    pl_module.train()
        else:
            return

        # Move images to CPU immediately to free GPU memory before video encoding
        images = images.detach().to("cpu")
        # Images: [B,3,H,W] or [B,3,T,H,W]
        videos = []
        if images.ndim == 4:
            # Make a dummy T=1 axis
            images = images[:, :, None, :, :]
        for i in range(images.shape[0]):
            vid = images[i : i + 1]  # [1,3,T,H,W]
            frames = _frames_from_video_afmhot(vid)
            # wandb.Video expects (T,H,W,3) np.uint8
            arr = np.stack(frames, axis=0)
            videos.append(wandb.Video(arr, fps=self.fps, format="gif"))

        # Log to W&B
        data = {self.log_key: videos}
        try:
            wandb.log(data, step=trainer.global_step)
        except Exception:
            pass
        finally:
            # Best-effort memory cleanup after sampling
            try:
                del images
                del videos
            except Exception:
                pass
            try:
                import gc
                gc.collect()
            except Exception:
                pass
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs, batch, batch_idx) -> None:  # type: ignore[override]
        self._maybe_sample_and_log(trainer, pl_module)


