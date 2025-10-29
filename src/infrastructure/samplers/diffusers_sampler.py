from __future__ import annotations

from typing import Optional, Tuple

import torch

from src.domain.interfaces.scheduler import Scheduler
from src.domain.interfaces.unet import UNetBackbone
from src.domain.interfaces.vae import VAE


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

        # Init noise latents scaled by scheduler's sigma
        latents = torch.randn(latents_shape, device=device, dtype=dtype)
        if hasattr(scheduler, "init_noise_sigma"):
            latents = latents * float(scheduler.init_noise_sigma)

        # Prepare timesteps
        timesteps = scheduler.set_timesteps(num_inference_steps=num_inference_steps, device=device)

        # Denoising loop
        use_amp = dtype in (torch.float16, torch.bfloat16)
        device_type = device.type if hasattr(device, "type") else ("cuda" if torch.cuda.is_available() else "cpu")
        amp_ctx = torch.autocast(device_type=device_type, dtype=dtype) if use_amp else _NullContext()
        with amp_ctx:
            for t in timesteps:
                model_input = scheduler.scale_model_input(sample=latents, timestep=t)
                model_output = unet(model_input, t, encoder_hidden_states=encoder_hidden_states)
                step_out = scheduler.step(model_output=model_output, timestep=t, sample=latents, return_dict=True)
                latents = step_out.prev_sample

        # Decode to images using VAE; its decode expects unscaled latents internally
        if latents.ndim == 4:
            images = vae.decode(latents)
            return images
        if latents.ndim == 5:
            # [B,C,F,H,W] -> [B*F,C,H,W] -> decode -> [B,F,3,H*8,W*8] -> [B,3,F,H*8,W*8]
            b, c, f, h, w = latents.shape
            latents_btchw = latents.permute(0, 2, 1, 3, 4).reshape(b * f, c, h, w)
            imgs_btchw = vae.decode(latents_btchw)
            imgs = imgs_btchw.reshape(b, f, imgs_btchw.shape[1], imgs_btchw.shape[2], imgs_btchw.shape[3]).permute(0, 2, 1, 3, 4)
            return imgs
        raise ValueError(f"Unsupported latents ndim: {latents.ndim}")


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


