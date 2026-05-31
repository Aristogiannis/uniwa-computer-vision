"""Numeric normalization for multispectral / RGB satellite imagery.

Satellite reflectance/radiance values span very different dynamic ranges
across sensors and bands, so naive min-max scaling can be wrecked by a single
saturated pixel. We therefore default to **percentile clipping** before
scaling — the same recipe used by ESA SNAP and most public Sentinel viewers.
"""

from __future__ import annotations

import numpy as np


def percentile_normalize(
    array: np.ndarray,
    *,
    lower: float = 2.0,
    upper: float = 98.0,
    per_channel: bool = True,
    eps: float = 1e-6,
) -> np.ndarray:
    """Clip to per-channel percentiles, then min-max scale into ``[0, 1]``.

    Parameters
    ----------
    array:
        Input array of shape ``(H, W)`` or ``(H, W, C)``.
    lower, upper:
        Percentile cut-offs in ``[0, 100]``.
    per_channel:
        Compute percentiles independently per channel (recommended for
        multispectral imagery). When ``False``, percentiles are global.
    eps:
        Numerical floor to avoid division by zero on flat tiles.
    """

    if not 0.0 <= lower < upper <= 100.0:
        raise ValueError("Expected 0 <= lower < upper <= 100, got "
                         f"lower={lower}, upper={upper}")

    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 2:
        lo = np.percentile(arr, lower)
        hi = np.percentile(arr, upper)
        return np.clip((arr - lo) / max(hi - lo, eps), 0.0, 1.0)

    if arr.ndim != 3:
        raise ValueError(f"Expected (H, W) or (H, W, C) array, got shape {arr.shape}")

    if per_channel:
        lo = np.percentile(arr, lower, axis=(0, 1), keepdims=True)
        hi = np.percentile(arr, upper, axis=(0, 1), keepdims=True)
    else:
        lo = np.percentile(arr, lower)
        hi = np.percentile(arr, upper)
    out = (arr - lo) / np.maximum(hi - lo, eps)
    return np.clip(out, 0.0, 1.0)


def minmax_normalize(array: np.ndarray, *, eps: float = 1e-6) -> np.ndarray:
    """Simple global min-max scaling to ``[0, 1]``."""

    arr = np.asarray(array, dtype=np.float32)
    return (arr - arr.min()) / max(arr.max() - arr.min(), eps)


def to_minus_one_one(array: np.ndarray) -> np.ndarray:
    """Map a ``[0, 1]`` array to the ``[-1, 1]`` range used by SD's VAE."""

    arr = np.asarray(array, dtype=np.float32)
    return arr * 2.0 - 1.0
