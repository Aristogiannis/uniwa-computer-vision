"""Fréchet Inception Distance using the ``clean-fid`` library.

We use :pypi:`clean-fid` (Parmar et al. CVPR 2022) — the de-facto reference
implementation that removes the resizing/quantization inconsistencies that
plague older FID pipelines and gives the numbers most reviewers expect.

Two modes are exposed:

* ``"clean"`` — the recommended Parmar-CVPR-2022 setup. Use this for the
  headline number reported in the paper.
* ``"legacy_pytorch"`` — bitwise-equivalent to ``pytorch-fid`` and matches
  most older satellite-diffusion papers. Use it only when explicitly
  comparing to those baselines.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from cv_diffusion.utils.logging import get_logger

logger = get_logger(__name__)

FidMode = Literal["clean", "legacy_pytorch", "legacy_tensorflow"]


def compute_fid(
    real_dir: str | Path,
    fake_dir: str | Path,
    *,
    mode: FidMode = "clean",
    batch_size: int = 32,
    num_workers: int = 2,
    device: str | None = None,
) -> float:
    """Compute FID between two folders of images.

    Both folders should contain only RGB PNG/JPG images; subdirectories are
    flattened by clean-fid.

    Returns the scalar FID. Lower is better; meaningful trends are usually
    differences ≥ 1.0.
    """

    from cleanfid import fid

    real = Path(real_dir)
    fake = Path(fake_dir)
    if not real.exists():
        raise FileNotFoundError(real)
    if not fake.exists():
        raise FileNotFoundError(fake)

    logger.info("Computing %s FID: real=%s fake=%s", mode, real, fake)
    score = fid.compute_fid(
        str(real),
        str(fake),
        mode=mode,
        num_workers=num_workers,
        batch_size=batch_size,
        device=device,
    )
    logger.info("FID(%s) = %.4f", mode, score)
    return float(score)
