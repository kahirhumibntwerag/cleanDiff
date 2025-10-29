from __future__ import annotations

import lightning.pytorch as pl
import torch
from torch.utils.data import Dataset, DataLoader

from src.application.usecases.without_encoder_module import TrainWithoutEncoderDiffusionModule
from src.infrastructure.encoders import create_speed_encoder

from tests.fakes import FakeOptimizerBuilder, FakeSampler, FakeScheduler, FakeUNet, FakeVAE


class _TinySpeedEhsDataset(Dataset):
    def __init__(self, *, encoder: torch.nn.Module, num_samples: int = 8, image_size: int = 32) -> None:
        self._n = num_samples
        self._s = image_size
        self._enc = encoder

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int):
        img = torch.randn(3, self._s, self._s)
        speed = torch.tensor([120.0], dtype=torch.float32)
        ehs = self._enc({"speed": speed}, return_dict=False)
        return {"pixel_values": img, "encoder_hidden_states": ehs}


def test_trainer_fit_with_fakes_runs_one_step() -> None:
    pl.seed_everything(123)
    encoder = create_speed_encoder(hidden_size=16)
    ds = _TinySpeedEhsDataset(encoder=encoder, num_samples=4, image_size=32)
    loader = DataLoader(ds, batch_size=2, shuffle=False)

    vae = FakeVAE(latent_channels=4, scaling_factor=0.5)
    unet = FakeUNet(in_channels=4, out_channels=4)
    scheduler = FakeScheduler()
    opt_builder = FakeOptimizerBuilder()
    sampler = FakeSampler()

    module = TrainWithoutEncoderDiffusionModule(
        vae=vae,
        unet=unet,
        scheduler=scheduler,
        optimizer_builder=opt_builder,
        sampler=sampler,
    )

    trainer = pl.Trainer(
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
    )
    trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)


