from __future__ import annotations

import argparse
from dataclasses import asdict
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from src.application.config import (
    SystemConfig,
    build_components,
    default_config,
    apply_mapping_overrides,
    apply_dot_overrides,
    env_overrides,
    load_config_file,
)
from src.application.usecases.without_encoder_module import TrainWithoutEncoderDiffusionModule
from src.application.usecases.with_encoder_module import TrainEncoderDiffusionModule
from src.infrastructure.datamodules import create_clip_video_datamodule
from src.infrastructure.callbacks.wandb_sampling import WandbVideoSamplingCallback


class _RandomDiffusionDataset(Dataset):
    def __init__(self, *, num_samples: int, image_size: int) -> None:
        super().__init__()
        self._num_samples = num_samples
        self._image_size = image_size

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, index: int):
        return {"pixel_values": torch.randn(3, self._image_size, self._image_size)}


def _precision_flag(precision: str) -> str | int:
    if precision == "float16":
        return "16-mixed"
    if precision == "bfloat16":
        return "bf16-mixed"
    return 32


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(description="Train diffusion UNet using factories and centralized config", add_help=True)
    # Training overrides
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--num-inference-steps", type=int, default=None)
    p.add_argument("--accelerator", type=str, default="auto")
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--precision", type=str, default=None, choices=["float32", "float16", "bfloat16"])
    p.add_argument("--seed", type=int, default=None)
    # Data
    p.add_argument("--train-root", type=str, default=None, help="Root folder containing clip_* subfolders and speeds.csv")
    p.add_argument("--val-root", type=str, default=None)
    p.add_argument("--num-frames", type=int, default=8)
    p.add_argument("--center-crop", action="store_true")
    # Use case
    p.add_argument("--usecase", type=str, choices=["no-encoder", "with-encoder"], default="with-encoder")
    p.add_argument("--enable-encoder", action="store_true")
    # W&B
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default="lightning-diffusion")
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument("--sample-every", type=int, default=200)
    p.add_argument("--sample-frames", type=int, default=8)
    p.add_argument("--sample-steps", type=int, default=25)
    p.add_argument("--sample-speeds", type=str, default="100,200,300")
    # Checkpointing
    p.add_argument("--ckpt-dir", type=str, default=None, help="Directory to save checkpoints")
    p.add_argument("--resume-from", type=str, default=None, help="Path to checkpoint to resume from")
    # Config system
    p.add_argument("--config", type=str, default=None, help="Path to config file (.json or .toml)")
    p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override any config field via dotted path, e.g. training.batch_size=8. Repeatable.",
    )
    return p.parse_known_args()


def _collect_unknown_overrides(unknown: list[str]) -> dict[str, str]:
    # Support formats: --a.b=c and pairs: --a.b c
    out: dict[str, str] = {}
    i = 0
    while i < len(unknown):
        tok = unknown[i]
        if tok.startswith("--"):
            body = tok[2:]
            if "=" in body:
                key, val = body.split("=", 1)
                out[key] = val
                i += 1
                continue
            # pair form
            if (i + 1) < len(unknown) and not unknown[i + 1].startswith("--"):
                out[body] = unknown[i + 1]
                i += 2
                continue
        i += 1
    return out


def main() -> None:
    args, unknown = parse_args()

    cfg = default_config()

    # File-based overrides
    if getattr(args, "config", None):
        mapping = load_config_file(args.config)
        apply_mapping_overrides(cfg, mapping)

    # Env overrides (prefix LIGHTNING_)
    apply_dot_overrides(cfg, env_overrides(prefix="LIGHTNING_"))

    # Free-form dotlist overrides from --set and unknown tokens
    dot_overrides: dict[str, str] = {}
    for kv in getattr(args, "set", []) or []:
        if "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        dot_overrides[k] = v
    dot_overrides.update(_collect_unknown_overrides(unknown))
    if dot_overrides:
        apply_dot_overrides(cfg, dot_overrides)

    # Apply CLI overrides
    if args.max_epochs is not None:
        cfg.training.max_epochs = args.max_epochs
    if args.batch_size is not None:
        cfg.training.batch_size = args.batch_size
    if args.image_size is not None:
        cfg.training.image_size = args.image_size
    if args.num_inference_steps is not None:
        cfg.training.num_inference_steps = args.num_inference_steps
    if args.precision is not None:
        cfg.training.precision = args.precision
    if args.seed is not None:
        cfg.training.seed = args.seed

    pl.seed_everything(cfg.training.seed, workers=True)

    components = build_components(cfg)

    # Optionally enable encoder from CLI
    if args.enable_encoder:
        cfg.encoder_enabled = True
        components = build_components(cfg)

    # Build module depending on use case
    if args.usecase == "with-encoder":
        assert components.get("encoder") is not None, "Encoder requested but not enabled/constructed in config"
        module = TrainEncoderDiffusionModule(
            vae=components["vae"],
            unet=components["unet"],
            scheduler=components["scheduler"],
            encoder=components["encoder"],
            optimizer_builder=components["optimizer_builder"],
            sampler=components["sampler"],
        )
    else:
        module = TrainWithoutEncoderDiffusionModule(
            vae=components["vae"],
            unet=components["unet"],
            scheduler=components["scheduler"],
            optimizer_builder=components["optimizer_builder"],
            sampler=components["sampler"],
        )

    # Data
    if args.train_root is not None:
        dm = create_clip_video_datamodule(
            train_root=args.train_root,
            val_root=args.val_root or args.train_root,
            batch_size=cfg.training.batch_size,
            num_workers=cfg.training.num_workers,
            resolution=cfg.training.image_size,
            num_frames=args.num_frames,
            center_crop=args.center_crop,
        )
        dm.setup()
        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()
    else:
        train_ds = _RandomDiffusionDataset(num_samples=128, image_size=cfg.training.image_size)
        val_ds = _RandomDiffusionDataset(num_samples=16, image_size=cfg.training.image_size)
        train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, num_workers=cfg.training.num_workers, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size, num_workers=cfg.training.num_workers)

    callbacks = []
    if args.wandb:
        try:
            import wandb as _wandb  # noqa: F401
            speeds = [float(x) for x in (args.sample_speeds.split(",") if args.sample_speeds else [])]
            callbacks.append(
                WandbVideoSamplingCallback(
                    sample_every_n_steps=args.sample_every,
                    num_inference_steps=args.sample_steps,
                    num_frames=args.sample_frames,
                    image_size=cfg.training.image_size,
                    speeds=speeds,
                )
            )
            logger = WandbLogger(project=args.wandb_project, name=args.wandb_run_name)
        except Exception:
            logger = False
    else:
        logger = False

    # Checkpoint callback
    if getattr(args, "ckpt_dir", None):
        callbacks.append(
            ModelCheckpoint(
                dirpath=args.ckpt_dir,
                save_last=True,
                save_top_k=1,
                monitor=None,
                filename="epoch{epoch:02d}-step{step}",
                every_n_epochs=1,
            )
        )

    trainer = pl.Trainer(
        accelerator="auto",
        devices=args.devices,
        max_epochs=cfg.training.max_epochs,
        max_steps=args.max_steps,
        precision=_precision_flag(cfg.training.precision),
        log_every_n_steps=10,
        callbacks=callbacks,
        logger=logger,
    )

    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=getattr(args, "resume_from", None))


if __name__ == "__main__":
    main()


