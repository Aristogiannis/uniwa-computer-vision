"""Spectral alignment helpers for multispectral satellite imagery.

The pretrained Stable Diffusion VAE operates on RGB. When fine-tuning on
multispectral data (e.g. Sentinel-2, which has 13 bands) we must pick a
3-channel composition that preserves the disaster-relevant signal.

For natural-disaster work we expose three useful compositions:

* ``rgb``       — natural-colour preview (B04, B03, B02)
* ``swir``      — short-wave infrared, ideal for burnt areas (B12, B11, B04)
* ``nrg``       — near-infrared / red / green, classic vegetation index (B08, B04, B03)

The dictionary ``SENTINEL2_BANDS`` lets callers reference bands by symbolic
name without hard-coding indices throughout the codebase.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

# 0-indexed band positions in a stack-of-13 SEN2 product.
SENTINEL2_BANDS: Mapping[str, int] = {
    "B01": 0,
    "B02": 1,
    "B03": 2,
    "B04": 3,
    "B05": 4,
    "B06": 5,
    "B07": 6,
    "B08": 7,
    "B8A": 8,
    "B09": 9,
    "B10": 10,
    "B11": 11,
    "B12": 12,
}

RGB_BAND_ORDER = ("B04", "B03", "B02")
SWIR_BAND_ORDER = ("B12", "B11", "B04")
NRG_BAND_ORDER = ("B08", "B04", "B03")


def select_bands(stack: np.ndarray, order: Sequence[str]) -> np.ndarray:
    """Return a ``(H, W, 3)`` array stacked from the requested bands.

    ``stack`` is expected in ``(H, W, C)`` or ``(C, H, W)`` layout — both are
    accepted to play nicely with rasterio / PIL conventions.
    """

    if stack.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {stack.shape}")

    if stack.shape[0] in (10, 12, 13) and stack.shape[2] not in (10, 12, 13):
        stack = np.transpose(stack, (1, 2, 0))

    missing = [b for b in order if b not in SENTINEL2_BANDS]
    if missing:
        raise KeyError(f"Unknown Sentinel-2 band(s): {missing}")

    idx = [SENTINEL2_BANDS[b] for b in order]
    max_idx = max(idx)
    if max_idx >= stack.shape[2]:
        raise IndexError(
            f"Band index {max_idx} out of range for stack with {stack.shape[2]} channels."
        )
    return stack[..., idx]


def rgb_from_sentinel2(stack: np.ndarray, composition: str = "rgb") -> np.ndarray:
    """Return a 3-channel composite suitable for SD's RGB pipeline."""

    composition = composition.lower()
    if composition == "rgb":
        order = RGB_BAND_ORDER
    elif composition == "swir":
        order = SWIR_BAND_ORDER
    elif composition == "nrg":
        order = NRG_BAND_ORDER
    else:
        raise ValueError(
            f"Unknown composition '{composition}'. Use one of: rgb, swir, nrg."
        )
    return select_bands(stack, order)
