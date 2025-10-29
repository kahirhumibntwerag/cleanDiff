from __future__ import annotations

import torch

from src.infrastructure.encoders.speed_encoder import SpeedEncoder


def test_speed_encoder_preserves_input_dtype_and_avoids_mismatch_cpu_bf16() -> None:
    # Use bfloat16 which is widely supported on CPU to simulate mixed-precision inputs
    dtype = torch.bfloat16
    enc = SpeedEncoder(hidden_size=16)

    # parameters default to float32; create inputs in bf16
    speeds = torch.tensor([100.0, 200.0, 300.0], dtype=dtype)

    out = enc({"speed": speeds}, return_dict=False)
    assert isinstance(out, torch.Tensor)
    # Output should match the input dtype for downstream compatibility
    assert out.dtype == dtype
    # Shape: [B, 1, D]
    assert out.shape == (speeds.shape[0], 1, enc.hidden_size)


