from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import csv
import re

import numpy as np
import torch
from PIL import Image

from src.domain.interfaces.types import DiffusionBatch


_CLIP_DIR_RE = re.compile(r"^clip_\d{5}$")
_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@dataclass
class ClipFrameDatasetConfig:
    root_dir: str
    speeds_csv: Optional[str] = None  # defaults to f"{root_dir}/speeds.csv"
    image_size: Optional[int] = None  # if provided, resize shortest side to this size keeping aspect, then center-crop square
    center_crop: bool = True
    frame_stride: int = 1            # take every k-th frame
    extensions: Sequence[str] = (".png", ".jpg", ".jpeg")


class ClipFrameDataset:
    """
    Dataset that treats each frame in clip folders as an individual diffusion training sample.
    - Directory layout:
        root_dir/
          speeds.csv          # CSV with columns: clip_id,speed  (clip_id like clip_00000)
          clip_00000/
            000000.png
            000001.png
            ...
          clip_00001/
            ...
    - Each sample returns DiffusionBatch with:
        pixel_values: float32 tensor [3,H,W] normalized to [-1, 1]
        encoder_inputs: {"speed": tensor([speed], dtype=float32)}
    """

    def __init__(self, cfg: ClipFrameDatasetConfig) -> None:
        self.cfg = cfg
        self.root = Path(cfg.root_dir)
        self.speeds_csv = Path(cfg.speeds_csv) if cfg.speeds_csv is not None else self.root / "speeds.csv"
        self._clip_to_speed: Dict[str, float] = self._load_speeds(self.speeds_csv)
        self._samples: List[tuple[Path, float]] = self._index_frames(self.root, set(ext.lower() for ext in cfg.extensions))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> DiffusionBatch:
        img_path, speed = self._samples[index]
        img = Image.open(img_path).convert("RGB")
        img = self._resize_and_crop(img)
        arr = np.array(img, dtype=np.uint8)
        tensor = torch.from_numpy(arr).permute(2, 0, 1).to(dtype=torch.float32) / 255.0
        tensor = tensor * 2.0 - 1.0  # normalize to [-1, 1]
        batch: DiffusionBatch = {
            "pixel_values": tensor,
            "encoder_inputs": {"speed": torch.tensor([speed], dtype=torch.float32)},
        }
        return batch

    # Collate is optional; default DataLoader works if returning consistent mapping, but provide for clarity
    @staticmethod
    def collate_fn(batch: Sequence[DiffusionBatch]) -> DiffusionBatch:
        pixel_values = torch.stack([b["pixel_values"] for b in batch], dim=0)
        speeds = torch.stack([b["encoder_inputs"]["speed"] for b in batch], dim=0)
        return {"pixel_values": pixel_values, "encoder_inputs": {"speed": speeds}}

    def _resize_and_crop(self, img: Image.Image) -> Image.Image:
        if self.cfg.image_size is None:
            return img
        size = self.cfg.image_size
        w, h = img.size
        # Resize shortest side to size, keep aspect
        if min(w, h) != size:
            if w < h:
                new_w = size
                new_h = int(round(h * (size / w)))
            else:
                new_h = size
                new_w = int(round(w * (size / h)))
            img = img.resize((new_w, new_h), Image.BICUBIC)
            w, h = img.size
        if self.cfg.center_crop:
            left = max((w - size) // 2, 0)
            top = max((h - size) // 2, 0)
            img = img.crop((left, top, left + size, top + size))
        else:
            # If not cropping, pad to square
            if w != size or h != size:
                pad_w = max(size - w, 0)
                pad_h = max(size - h, 0)
                pad_left = pad_w // 2
                pad_top = pad_h // 2
                new_img = Image.new("RGB", (max(w, size), max(h, size)), (0, 0, 0))
                new_img.paste(img, (pad_left, pad_top))
                img = new_img
        return img

    def _load_speeds(self, csv_path: Path) -> Dict[str, float]:
        mapping: Dict[str, float] = {}
        if not csv_path.exists():
            raise FileNotFoundError(f"Speeds CSV not found: {csv_path}")
        with csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            # Expect columns: clip_id,speed
            for row in reader:
                clip_id = (row.get("clip_id") or row.get("clip") or row.get("id") or "").strip()
                if not clip_id:
                    continue
                speed_str = row.get("speed") or row.get("speeds") or row.get("value")
                if speed_str is None:
                    continue
                try:
                    speed = float(speed_str)
                except ValueError:
                    continue
                mapping[clip_id] = speed
        if not mapping:
            raise ValueError(f"No speeds found in {csv_path}; expected header with 'clip_id,speed'")
        return mapping

    def _index_frames(self, root: Path, exts: set[str]) -> List[tuple[Path, float]]:
        samples: List[tuple[Path, float]] = []
        if not root.exists():
            raise FileNotFoundError(f"Root directory not found: {root}")
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            if not _CLIP_DIR_RE.match(child.name):
                continue
            speed = self._clip_to_speed.get(child.name)
            if speed is None:
                # allow folders without a speed row to be skipped
                continue
            frame_paths = [p for p in sorted(child.iterdir()) if p.suffix.lower() in exts]
            if self.cfg.frame_stride > 1:
                frame_paths = frame_paths[:: self.cfg.frame_stride]
            for fp in frame_paths:
                samples.append((fp, speed))
        if not samples:
            raise ValueError(f"No frames indexed under {root} using pattern 'clip_00000' and extensions {sorted(exts)}")
        return samples


