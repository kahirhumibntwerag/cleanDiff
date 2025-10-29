from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule

from src.domain.interfaces import Scheduler, UNetBackbone, VAE, Sampler
from src.domain.interfaces.types import DiffusionBatch
from src.domain.interfaces.optimizer import OptimizerBuilder


class TrainWithoutEncoderDiffusionModule(LightningModule):
    """
    Use-case: train UNet only (and optionally VAE) without training an encoder.
    Batch may include `encoder_hidden_states` precomputed; otherwise unconditioned.
    """

    def __init__(
        self,
        *,
        vae: VAE,
        unet: UNetBackbone,
        scheduler: Scheduler,
        optimizer_builder: OptimizerBuilder,
        sampler: Sampler,
    ) -> None:
        super().__init__()
        self.vae = vae
        self.unet = unet
        self.scheduler = scheduler
        self.save_hyperparameters({"train_vae": False, "train_encoder": False})
        # VAE is always frozen in this use case
        self._set_requires_grad(self.vae, False)
        self.optimizer_builder = optimizer_builder
        self.sampler = sampler

    @staticmethod
    def _set_requires_grad(module: torch.nn.Module, requires_grad: bool) -> None:
        if hasattr(module, "parameters") and callable(getattr(module, "parameters")):
            for p in module.parameters():
                p.requires_grad = requires_grad

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
        with torch.no_grad():
            enc = self.vae.encode(images, sample=sample_from_posterior)
        return enc.latents * self.vae.scaling_factor

    def training_step(self, batch: DiffusionBatch, batch_idx: int) -> torch.Tensor:
        images: torch.Tensor = batch["pixel_values"]
        encoder_hidden_states = batch.get("encoder_hidden_states")  # type: ignore[assignment]
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
        return loss

    def validation_step(self, batch: DiffusionBatch, batch_idx: int) -> torch.Tensor:
        images: torch.Tensor = batch["pixel_values"]
        encoder_hidden_states = batch.get("encoder_hidden_states")  # type: ignore[assignment]
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
        return optimizers[0]

    @torch.no_grad()
    def sample(
        self,
        *,
        num_inference_steps: int,
        latents_shape: Tuple[int, int, int, int],
        encoder_hidden_states: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
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
