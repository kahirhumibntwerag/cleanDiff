from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import csv
import json

import numpy as np
import torch
from PIL import Image

from src.domain.interfaces.types import DiffusionBatch


@dataclass
class ClipVideoDatasetConfig:
    root_dir: str
    speeds_csv: Optional[str] = None
    resolution: int = 256
    num_frames: int = 8
    center_crop: bool = True
    limit_first_n: Optional[int] = None


class ClipVideoDataset:
    """
    Loads clips from folders named by clip_id (e.g., clip_00000) under `root_dir` and
    their speeds from speeds.csv with columns: clip_id,speed

    Returns DiffusionBatch per item with:
      - pixel_values: FloatTensor [3, T, H, W] in [-1, 1]
      - encoder_inputs: {"speed": FloatTensor([speed])}
    """

    def __init__(self, cfg: ClipVideoDatasetConfig) -> None:
        self.cfg = cfg
        self.root = Path(cfg.root_dir)
        csv_path = Path(cfg.speeds_csv) if cfg.speeds_csv is not None else self.root / "speeds.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"speeds.csv not found at: {csv_path}")

        rows: List[tuple[str, float]] = []
        with csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            if not {"clip_id", "speed"}.issubset(set(reader.fieldnames or [])):
                raise ValueError("speeds.csv must contain columns: clip_id,speed")
            for row in reader:
                cid = str(row["clip_id"]).strip()
                sp = float(row["speed"])  # raises on bad data
                rows.append((cid, sp))

        if cfg.limit_first_n is not None and cfg.limit_first_n > 0:
            rows = rows[: cfg.limit_first_n]

        self.items = rows
        speeds = np.asarray([s for _, s in self.items], dtype=np.float32)
        self.mean: float = float(speeds.mean())
        self.std: float = float(speeds.std() + 1e-6)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> DiffusionBatch:
        clip_id, speed = self.items[idx]
        clip_dir = self.root / clip_id
        if not clip_dir.exists():
            raise FileNotFoundError(f"clip folder not found: {clip_dir}")

        frames = []
        T = int(self.cfg.num_frames)
        res = int(self.cfg.resolution)
        for k in range(T):
            fp = clip_dir / f"{k:03d}.png"
            if not fp.exists():
                fp = fp.with_suffix(".jpg")
            if not fp.exists():
                raise FileNotFoundError(f"missing frame for {clip_id}: {fp}")
            img = Image.open(fp).convert("RGB")
            img = img.resize((res, res), Image.LANCZOS)
            if self.cfg.center_crop:
                # Already square; kept for parity with provided code
                pass
            arr = np.array(img, dtype=np.uint8)
            ten = torch.from_numpy(arr).permute(2, 0, 1).to(dtype=torch.float32) / 255.0
            ten = ten * 2.0 - 1.0
            frames.append(ten)

        video = torch.stack(frames, dim=1)  # [3, T, H, W]
        batch: DiffusionBatch = {
            "pixel_values": video,
            "encoder_inputs": {"speed": torch.tensor([speed], dtype=torch.float32)},
        }
        return batch

    @staticmethod
    def collate_fn(batch: Sequence[DiffusionBatch]) -> DiffusionBatch:
        pixel_values = torch.stack([b["pixel_values"] for b in batch], dim=0)
        speeds = torch.stack([b["encoder_inputs"]["speed"] for b in batch], dim=0)
        return {"pixel_values": pixel_values, "encoder_inputs": {"speed": speeds}}


