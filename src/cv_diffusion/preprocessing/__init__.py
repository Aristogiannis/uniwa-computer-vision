"""Satellite image preprocessing utilities.

Implements the preprocessing methodology described in the project report:

* normalization (percentile clipping + 0..1 / -1..1 scaling)
* tiling (sliding window crops with configurable stride/overlap)
* spectral alignment (Sentinel-1/-2 band selection and RGB composition)

The torch-dependent dataset classes live in :mod:`cv_diffusion.preprocessing.dataset`
and are imported lazily on first attribute access so that ``import
cv_diffusion.preprocessing`` (or its sibling modules) does not pull in
PyTorch when a caller only needs ``normalize`` / ``tile`` / ``spectral``.
"""

from cv_diffusion.preprocessing.normalize import (
    minmax_normalize,
    percentile_normalize,
    to_minus_one_one,
)
from cv_diffusion.preprocessing.spectral import (
    SENTINEL2_BANDS,
    rgb_from_sentinel2,
    select_bands,
)
from cv_diffusion.preprocessing.tile import (
    TileSpec,
    iter_tiles,
    tile_image,
)

from cv_diffusion.preprocessing.prompts import (
    DISASTER_PROMPT_TEMPLATES,
    build_text_prompt,
)

_LAZY_NAMES = {
    "DisasterImageDataset": "cv_diffusion.preprocessing.dataset",
    "SatelliteFolderDataset": "cv_diffusion.preprocessing.dataset",
}


def __getattr__(name: str):  # noqa: D401
    target = _LAZY_NAMES.get(name)
    if target is None:
        raise AttributeError(f"module 'cv_diffusion.preprocessing' has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(target), name)


__all__ = [
    "DISASTER_PROMPT_TEMPLATES",
    "DisasterImageDataset",
    "SENTINEL2_BANDS",
    "SatelliteFolderDataset",
    "TileSpec",
    "build_text_prompt",
    "iter_tiles",
    "minmax_normalize",
    "percentile_normalize",
    "rgb_from_sentinel2",
    "select_bands",
    "tile_image",
    "to_minus_one_one",
]
