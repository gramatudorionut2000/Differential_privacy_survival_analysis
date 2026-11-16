from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """

    this does *not* make DP-SGD bit-for-bit reproducible: Opacus draws its
    Gaussian noise from the torch RNG, so seeding fixes the noise draw, but
    Poisson subsampling batch sizes still vary run to run by design.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(prefer: str = "auto") -> torch.device:
    if prefer == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(prefer)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p




def steps_per_epoch(n: int, batch_size: int) -> int:
    """Optimisation steps in one epoch = batches Opacus' DPDataLoader yields."""
    return max(int(np.ceil(n / batch_size)), 1)


def sampling_shape(n: int, batch_size: int, epochs: int) -> tuple[float, int]:
    """``(sample_rate, total_steps)`` to hand the privacy accountant.

    The sample rate is ``1 / steps_per_epoch``.
    Opacus builds its Poisson sampler with ``sample_rate = 1 / len(loader)``, and
    since the loader yields ``ceil(n / batch_size)`` batches those two differ
    whenever ``n`` is not a multiple of ``batch_size``.
    """
    spe = steps_per_epoch(n, batch_size)
    return 1.0 / spe, epochs * spe
