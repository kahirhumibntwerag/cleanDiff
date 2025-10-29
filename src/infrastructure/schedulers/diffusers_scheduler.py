from __future__ import annotations

from typing import Optional

import torch

from diffusers import (
    DDPMScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    HeunDiscreteScheduler,
    DPMSolverMultistepScheduler,
    UniPCMultistepScheduler,
)

from src.domain.interfaces.types import PredictionType, SchedulerStepOutput


_PRED_MAP = {
    "epsilon": PredictionType.EPSILON,
    "v_prediction": PredictionType.V_PREDICTION,
    "sample": PredictionType.SAMPLE,
}


class DiffusersSchedulerAdapter:
    """
    Adapter wrapping a Diffusers scheduler instance to conform to our Scheduler protocol.
    """

    def __init__(self, scheduler: object) -> None:
        self.scheduler = scheduler

    # ----- Properties -----
    @property
    def num_train_timesteps(self) -> int:
        val = getattr(getattr(self.scheduler, "config", object()), "num_train_timesteps", None)
        if isinstance(val, int):
            return val
        return int(getattr(self.scheduler, "num_train_timesteps", 1000))

    @property
    def init_noise_sigma(self) -> float:
        return float(getattr(self.scheduler, "init_noise_sigma", 1.0))

    @property
    def prediction_type(self) -> PredictionType:
        raw = getattr(getattr(self.scheduler, "config", object()), "prediction_type", "epsilon")
        return _PRED_MAP.get(str(raw), PredictionType.EPSILON)

    # ----- Methods -----
    def set_timesteps(self, num_inference_steps: int, device: torch.device) -> torch.Tensor:
        # Some schedulers accept device kwarg, others infer. Try to pass device.
        try:
            self.scheduler.set_timesteps(num_inference_steps=num_inference_steps, device=device)
        except TypeError:
            self.scheduler.set_timesteps(num_inference_steps)
        timesteps = getattr(self.scheduler, "timesteps", None)
        if timesteps is None:
            # Some older schedulers return the timesteps
            try:
                timesteps = self.scheduler.set_timesteps(num_inference_steps)
            except Exception:
                timesteps = torch.arange(num_inference_steps, device=device)
        return timesteps.to(device)

    def scale_model_input(self, sample: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        fn = getattr(self.scheduler, "scale_model_input", None)
        if callable(fn):
            # Some schedulers (e.g., EulerDiscreteScheduler) expect a scalar timestep matching
            # their internal schedule, not a batch-sized vector. Coerce batched timesteps to scalar.
            ts = timestep
            if isinstance(ts, torch.Tensor) and ts.ndim > 0:
                ts = ts[0]
            return fn(sample=sample, timestep=ts)
        return sample

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        return_dict: bool = True,
    ) -> SchedulerStepOutput:
        out = self.scheduler.step(model_output=model_output, timestep=timestep, sample=sample, return_dict=True)
        prev = getattr(out, "prev_sample", None)
        x0 = getattr(out, "pred_original_sample", None)
        return SchedulerStepOutput(prev_sample=prev, pred_original_sample=x0)

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        fn = getattr(self.scheduler, "add_noise", None)
        if callable(fn):
            try:
                return fn(original_samples=original_samples, noise=noise, timesteps=timesteps)
            except Exception:
                # Some schedulers (e.g., EulerDiscreteScheduler) expect timesteps from the internal schedule.
                # Map integer indices -> scheduled timesteps and retry.
                schedule = getattr(self.scheduler, "timesteps", None)
                if schedule is None or int(getattr(schedule, "numel", lambda: 0)()) == 0:
                    # Initialize schedule with the full training horizon
                    device = original_samples.device
                    try:
                        self.set_timesteps(num_inference_steps=self.num_train_timesteps, device=device)
                        schedule = getattr(self.scheduler, "timesteps", None)
                    except Exception:
                        schedule = None
                if schedule is not None and int(getattr(schedule, "numel", lambda: 0)()) > 0:
                    if isinstance(timesteps, torch.Tensor):
                        # Convert to indices if dtype is integral; otherwise, assume already on schedule
                        if timesteps.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.long, torch.short):
                            idx = timesteps.clamp(0, schedule.shape[0] - 1).to(device=schedule.device, dtype=torch.long)
                            ts = schedule[idx]
                        else:
                            ts = timesteps.to(device=schedule.device, dtype=schedule.dtype)
                        if ts.ndim == 0:
                            ts = ts.unsqueeze(0)
                    else:
                        if isinstance(timesteps, int):
                            idx = max(0, min(int(timesteps), schedule.shape[0] - 1))
                            ts = schedule[idx].unsqueeze(0)
                        else:
                            ts = timesteps
                    return fn(original_samples=original_samples, noise=noise, timesteps=ts)
        # Fallback (not strictly correct for all schedulers)
        return original_samples + noise

    def get_velocity(
        self,
        sample: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        fn = getattr(self.scheduler, "get_velocity", None)
        if callable(fn):
            return fn(sample=sample, noise=noise, timesteps=timestep)
        return noise


# Convenience factory helpers (kept here for discoverability)
def _make_ddpm(**kwargs) -> DiffusersSchedulerAdapter:
    return DiffusersSchedulerAdapter(DDPMScheduler(**kwargs))


def _make_ddim(**kwargs) -> DiffusersSchedulerAdapter:
    return DiffusersSchedulerAdapter(DDIMScheduler(**kwargs))


def _make_euler(**kwargs) -> DiffusersSchedulerAdapter:
    return DiffusersSchedulerAdapter(EulerDiscreteScheduler(**kwargs))


def _make_euler_ancestral(**kwargs) -> DiffusersSchedulerAdapter:
    return DiffusersSchedulerAdapter(EulerAncestralDiscreteScheduler(**kwargs))


def _make_heun(**kwargs) -> DiffusersSchedulerAdapter:
    return DiffusersSchedulerAdapter(HeunDiscreteScheduler(**kwargs))


def _make_dpmpp_2m(**kwargs) -> DiffusersSchedulerAdapter:
    return DiffusersSchedulerAdapter(DPMSolverMultistepScheduler(**kwargs))


def _make_unipc(**kwargs) -> DiffusersSchedulerAdapter:
    return DiffusersSchedulerAdapter(UniPCMultistepScheduler(**kwargs))


