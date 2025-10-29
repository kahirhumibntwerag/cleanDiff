from __future__ import annotations

from typing import Optional, Tuple

import os
import logging

import torch

from src.domain.interfaces.scheduler import Scheduler
from src.domain.interfaces.unet import UNetBackbone
from src.domain.interfaces.vae import VAE


_logger = logging.getLogger(__name__)


def _debug_enabled() -> bool:
    val = os.environ.get("LIGHTNING_DEBUG_SAMPLER", "").strip().lower()
    return val not in ("", "0", "false", "no")


def _tensor_stats(t: torch.Tensor) -> dict:
    try:
        return {
            "min": float(t.min().item()),
            "max": float(t.max().item()),
            "mean": float(t.mean().item()),
            "std": float(t.std().item()),
        }
    except Exception:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}


class DiffusersDenoiserSampler:
    @torch.no_grad()
    def sample(
        self,
        *,
        unet: UNetBackbone,
        scheduler: Scheduler,
        vae: VAE,
        encoder_hidden_states: Optional[torch.Tensor],
        num_inference_steps: int,
        latents_shape: Tuple[int, int, int, int] | Tuple[int, int, int, int, int],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        # Resolve device/dtype
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if dtype is None:
            dtype = torch.float32

        if _debug_enabled():
            _logger.info(
                "[Sampler] start device=%s dtype=%s latents_shape=%s pred_type=%s",
                getattr(device, "type", str(device)),
                str(dtype),
                tuple(latents_shape),
                getattr(getattr(scheduler, "prediction_type", None), "value", getattr(scheduler, "prediction_type", None)),
            )

        # Init noise latents scaled by scheduler's sigma
        latents = torch.randn(latents_shape, device=device, dtype=dtype)
        if hasattr(scheduler, "init_noise_sigma"):
            latents = latents * float(scheduler.init_noise_sigma)
        if _debug_enabled():
            _logger.info(
                "[Sampler] init latents stats=%s init_noise_sigma=%s",
                _tensor_stats(latents),
                getattr(scheduler, "init_noise_sigma", None),
            )

        # Prepare timesteps
        timesteps = scheduler.set_timesteps(num_inference_steps=num_inference_steps, device=device)
        if _debug_enabled():
            try:
                nsteps = int(timesteps.shape[0])  # type: ignore[attr-defined]
                first = timesteps[0].item() if nsteps > 0 else None
                last = timesteps[-1].item() if nsteps > 0 else None
            except Exception:
                nsteps, first, last = 0, None, None
            _logger.info("[Sampler] timesteps count=%s first=%s last=%s", nsteps, first, last)

        # Denoising loop
        use_amp = dtype in (torch.float16, torch.bfloat16)
        device_type = device.type if hasattr(device, "type") else ("cuda" if torch.cuda.is_available() else "cpu")
        amp_ctx = torch.autocast(device_type=device_type, dtype=dtype) if use_amp else _NullContext()
        with amp_ctx:
            step_index = -1
            try:
                total_steps = int(timesteps.shape[0])  # type: ignore[attr-defined]
            except Exception:
                total_steps = 0
            for t in timesteps:
                step_index += 1
                model_input = scheduler.scale_model_input(sample=latents, timestep=t)
                model_output = unet(model_input, t, encoder_hidden_states=encoder_hidden_states)
                step_out = scheduler.step(model_output=model_output, timestep=t, sample=latents, return_dict=True)
                latents = step_out.prev_sample
                if _debug_enabled() and (step_index in (0, max(0, total_steps - 1))):
                    try:
                        _logger.info(
                            "[Sampler] step %d/%d model_input.std=%.6f latents.std=%.6f",
                            step_index + 1,
                            total_steps,
                            float(model_input.std().item()),
                            float(latents.std().item()),
                        )
                    except Exception:
                        pass

        # Decode to images using VAE; its decode expects unscaled latents internally
        # Ensure latents dtype matches VAE parameters to avoid dtype mismatch in layers/biases
        try:
            vae_param_dtype = next(vae.parameters()).dtype  # type: ignore[attr-defined]
        except Exception:
            vae_param_dtype = latents.dtype
        if _debug_enabled():
            _logger.info("[Sampler] before decode latents stats=%s", _tensor_stats(latents))
        if latents.ndim == 4:
            images = vae.decode(latents.to(dtype=vae_param_dtype))
            if _debug_enabled():
                _logger.info("[Sampler] decoded image stats=%s", _tensor_stats(images))
            return images
        if latents.ndim == 5:
            # [B,C,F,H,W] -> [B*F,C,H,W] -> decode -> [B,F,3,H*8,W*8] -> [B,3,F,H*8,W*8]
            b, c, f, h, w = latents.shape
            latents_btchw = latents.permute(0, 2, 1, 3, 4).reshape(b * f, c, h, w)
            imgs_btchw = vae.decode(latents_btchw.to(dtype=vae_param_dtype))
            imgs = imgs_btchw.reshape(b, f, imgs_btchw.shape[1], imgs_btchw.shape[2], imgs_btchw.shape[3]).permute(0, 2, 1, 3, 4)
            if _debug_enabled():
                # Compute stats on a flattened temporal batch for readability
                try:
                    imgs_stats = _tensor_stats(imgs_btchw)
                except Exception:
                    imgs_stats = {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
                _logger.info("[Sampler] decoded video frame stats=%s", imgs_stats)
            return imgs
        raise ValueError(f"Unsupported latents ndim: {latents.ndim}")


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


