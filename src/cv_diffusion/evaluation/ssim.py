"""SSIM helpers used in qualitative evaluation.

Two complementary numbers are reported:

* :func:`compute_pairwise_ssim` — average SSIM between matched (real, fake)
  pairs (when we have a 1:1 correspondence, e.g. paired pre/post tiles).
* :func:`compute_set_ssim`     — mean of the per-fake-image *nearest-real*
  SSIM across two unaligned sets. Useful when synthetic images have no
  natural pairing (the more common case here).

We delegate the actual computation to :pypi:`torchmetrics` because it runs
on the GPU and gives identical numbers to scikit-image up to floating point
noise.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from cv_diffusion.utils.io import list_images
from cv_diffusion.utils.logging import get_logger

logger = get_logger(__name__)


def _load_tensor_batch(paths: Sequence[Path], size: int):
    import torch
    from torchvision import transforms

    tfm = transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])
    tensors = [tfm(Image.open(p).convert("RGB")) for p in paths]
    return torch.stack(tensors)


def compute_pairwise_ssim(
    real_dir: str | Path,
    fake_dir: str | Path,
    *,
    image_size: int = 256,
    device: str | None = None,
) -> dict:
    """Mean SSIM over matched real/fake pairs.

    Pairs are formed by sorting filenames in each folder. This is intended
    for evaluations where the user has deliberately aligned outputs
    (e.g. generated 1 synthetic image per real reference).
    """

    import torch
    from torchmetrics.image import StructuralSimilarityIndexMeasure

    reals = list_images(real_dir)
    fakes = list_images(fake_dir)
    n = min(len(reals), len(fakes))
    if n == 0:
        raise ValueError("No images found in real_dir or fake_dir.")
    if len(reals) != len(fakes):
        logger.warning("Folder sizes differ (%d vs %d); truncating to %d.", len(reals), len(fakes), n)

    real_t = _load_tensor_batch(reals[:n], image_size)
    fake_t = _load_tensor_batch(fakes[:n], image_size)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    real_t, fake_t = real_t.to(dev), fake_t.to(dev)

    metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(dev)
    score = metric(fake_t, real_t)
    return {"mean_ssim": float(score.item()), "n_pairs": n}


def compute_set_ssim(
    real_dir: str | Path,
    fake_dir: str | Path,
    *,
    image_size: int = 256,
    max_real: int = 256,
    max_fake: int = 256,
    device: str | None = None,
) -> dict:
    """Mean of per-fake nearest-real SSIM, plus the mean of *all* pairs.

    Provides a rough sense of how close the synthetic distribution sits to
    the real one. Note: ``O(R*F)`` cost, so we cap both sides at
    ``max_real`` / ``max_fake`` images by default.
    """

    import torch
    from torchmetrics.functional.image import structural_similarity_index_measure as ssim_fn

    reals = list_images(real_dir)[:max_real]
    fakes = list_images(fake_dir)[:max_fake]
    if not reals or not fakes:
        raise ValueError("No images found in real_dir or fake_dir.")

    real_t = _load_tensor_batch(reals, image_size)
    fake_t = _load_tensor_batch(fakes, image_size)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    real_t, fake_t = real_t.to(dev), fake_t.to(dev)

    all_scores: list[float] = []
    nearest_scores: list[float] = []
    for f_idx in range(fake_t.shape[0]):
        f = fake_t[f_idx : f_idx + 1].expand(real_t.shape[0], -1, -1, -1)
        score = ssim_fn(f, real_t, data_range=1.0, reduction="none")
        score = score.view(-1)
        all_scores.extend(score.detach().cpu().tolist())
        nearest_scores.append(float(score.max().item()))

    return {
        "mean_ssim_all_pairs": float(np.mean(all_scores)),
        "mean_nearest_ssim": float(np.mean(nearest_scores)),
        "n_real": len(reals),
        "n_fake": len(fakes),
    }
