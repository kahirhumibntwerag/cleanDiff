from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule

from src.domain.interfaces import Encoder, Scheduler, UNetBackbone, VAE, Sampler
from src.domain.interfaces.types import DiffusionBatch, EncoderInputs
from src.domain.interfaces.optimizer import OptimizerBuilder


class TrainEncoderDiffusionModule(LightningModule):
    def __init__(
        self,
        *,
        vae: VAE,
        unet: UNetBackbone,
        scheduler: Scheduler,
        encoder: Encoder,
        optimizer_builder: OptimizerBuilder,
        sampler: Sampler,
    ) -> None:
        super().__init__()
        self.vae = vae
        self.unet = unet
        self.scheduler = scheduler
        self.encoder = encoder
        self.save_hyperparameters({"train_vae": False, "train_encoder": True})
        # VAE is always frozen in this use case
        self._set_requires_grad(self.vae, False)
        # Always train encoder in this use case
        self._set_requires_grad(self.encoder, True)
        self.optimizer_builder = optimizer_builder
        self.sampler = sampler
        # Multiple optimizers -> use manual optimization per Lightning docs
        self.automatic_optimization = False

    @staticmethod
    def _set_requires_grad(module: torch.nn.Module, requires_grad: bool) -> None:
        if hasattr(module, "parameters") and callable(getattr(module, "parameters")):
            for p in module.parameters():
                p.requires_grad = requires_grad

    def _extract_encoder_hidden_states(self, batch: DiffusionBatch) -> Optional[torch.Tensor]:
        ehs = batch.get("encoder_hidden_states")  # type: ignore[assignment]
        if isinstance(ehs, torch.Tensor):
            return ehs
        inputs = batch.get("encoder_inputs")  # type: ignore[assignment]
        if inputs is None:
            return None
        out = self.encoder(inputs, return_dict=False)  # type: ignore[arg-type]
        if isinstance(out, dict):
            return out.get("encoder_hidden_states")
        return out

    def _compute_target(self, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        pred_type = self.scheduler.prediction_type
        if pred_type.value == "epsilon":
            return noise
        if pred_type.value == "v_prediction":
            return self.scheduler.get_velocity(sample=latents, noise=noise, timestep=timesteps)
        if pred_type.value == "sample":
            return latents
        raise ValueError(f"Unsupported prediction_type: {pred_type}")

    def _prepare_latents(self, images: torch.Tensor, sample_from_posterior: bool) -> torch.Tensor:
        # Accept either images [B, C, H, W] or video [B, C, T, H, W]
        with torch.no_grad():
            if images.ndim == 5:
                b, c, t, h, w = images.shape
                flat = images.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
                enc = self.vae.encode(flat, sample=sample_from_posterior)
                lat = enc.latents.reshape(b, t, -1, enc.latents.shape[-2], enc.latents.shape[-1]).permute(0, 2, 1, 3, 4)
            else:
                enc = self.vae.encode(images, sample=sample_from_posterior)
                lat = enc.latents
        return lat * self.vae.scaling_factor

    def training_step(self, batch: DiffusionBatch, batch_idx: int) -> torch.Tensor:
        images: torch.Tensor = batch["pixel_values"]
        encoder_hidden_states = self._extract_encoder_hidden_states(batch)
        latents = self._prepare_latents(images, sample_from_posterior=True)
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (bsz,), device=latents.device, dtype=torch.long)
        noisy_latents = self.scheduler.add_noise(original_samples=latents, noise=noise, timesteps=timesteps)
        model_input = self.scheduler.scale_model_input(sample=noisy_latents, timestep=timesteps)
        model_pred = self.unet(model_input, timesteps, encoder_hidden_states=encoder_hidden_states)
        assert isinstance(model_pred, torch.Tensor), "UNetBackbone must return a torch.Tensor"
        target = self._compute_target(latents=latents, noise=noise, timesteps=timesteps)
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=bsz)

        # Manual optimization over both optimizers (UNet + Encoder) when attached to a Trainer
        if getattr(self, "_trainer", None) is not None:
            opt_unet, opt_enc = self.optimizers()
            opt_unet.zero_grad(set_to_none=True)
            opt_enc.zero_grad(set_to_none=True)
            self.manual_backward(loss)
            opt_unet.step()
            opt_enc.step()

        return loss

    def validation_step(self, batch: DiffusionBatch, batch_idx: int) -> torch.Tensor:
        images: torch.Tensor = batch["pixel_values"]
        encoder_hidden_states = self._extract_encoder_hidden_states(batch)
        latents = self._prepare_latents(images, sample_from_posterior=False)
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (bsz,), device=latents.device, dtype=torch.long)
        noisy_latents = self.scheduler.add_noise(original_samples=latents, noise=noise, timesteps=timesteps)
        model_input = self.scheduler.scale_model_input(sample=noisy_latents, timestep=timesteps)
        model_pred = self.unet(model_input, timesteps, encoder_hidden_states=encoder_hidden_states)
        assert isinstance(model_pred, torch.Tensor), "UNetBackbone must return a torch.Tensor"
        target = self._compute_target(latents=latents, noise=noise, timesteps=timesteps)
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=bsz)
        return loss

    def configure_optimizers(self):
        optimizers = []
        # UNet optimizer
        unet_spec = self.unet.optimizer_spec()
        optimizers.append(self.optimizer_builder.build(spec=unet_spec, params=self.unet.parameters()))
        # Encoder (always trained here)
        enc_spec = self.encoder.optimizer_spec()
        optimizers.append(self.optimizer_builder.build(spec=enc_spec, params=self.encoder.parameters()))
        return optimizers if len(optimizers) > 1 else optimizers[0]

    @torch.no_grad()
    def sample(
        self,
        *,
        num_inference_steps: int,
        latents_shape: Tuple[int, int, int, int],
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_inputs: Optional[EncoderInputs] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        # Resolve conditioning
        if encoder_hidden_states is None and encoder_inputs is not None:
            out = self.encoder(encoder_inputs, return_dict=False)  # type: ignore[arg-type]
            if isinstance(out, dict):
                encoder_hidden_states = out.get("encoder_hidden_states")
            else:
                encoder_hidden_states = out
        return self.sampler.sample(
            unet=self.unet,
            scheduler=self.scheduler,
            vae=self.vae,
            encoder_hidden_states=encoder_hidden_states,
            num_inference_steps=num_inference_steps,
            latents_shape=latents_shape,
            device=device,
            dtype=dtype,
        )
