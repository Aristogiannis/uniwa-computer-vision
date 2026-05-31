"""Sliding-window tiling for large satellite scenes (xBD / SEN12MS)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class TileSpec:
    """Configuration for sliding-window tiling.

    Attributes
    ----------
    size:
        Side length of each square tile in pixels.
    stride:
        Step between tile origins. Setting ``stride < size`` produces
        overlapping tiles (useful for augmenting small datasets).
    drop_partial:
        If ``True``, skip the trailing edge tiles that wouldn't fit. If
        ``False``, pad those tiles with reflection so the entire scene is
        covered.
    """

    size: int = 512
    stride: int = 512
    drop_partial: bool = True

    def __post_init__(self) -> None:
        if self.size <= 0 or self.stride <= 0:
            raise ValueError("size and stride must be positive integers.")


def iter_tiles(image: np.ndarray, spec: TileSpec) -> Iterator[tuple[int, int, np.ndarray]]:
    """Yield ``(y, x, tile)`` triples covering ``image`` per ``spec``."""

    if image.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D array, got shape {image.shape}")

    h, w = image.shape[:2]
    size, stride = spec.size, spec.stride

    ys = list(range(0, max(h - size, 0) + 1, stride))
    xs = list(range(0, max(w - size, 0) + 1, stride))

    if not spec.drop_partial:
        if not ys or ys[-1] + size < h:
            ys.append(max(h - size, 0))
        if not xs or xs[-1] + size < w:
            xs.append(max(w - size, 0))

    for y in ys:
        for x in xs:
            tile = image[y : y + size, x : x + size]
            th, tw = tile.shape[:2]
            if (th, tw) != (size, size):
                if spec.drop_partial:
                    continue
                pad_y = size - th
                pad_x = size - tw
                pad_width = ((0, pad_y), (0, pad_x))
                if image.ndim == 3:
                    pad_width = (*pad_width, (0, 0))
                tile = np.pad(tile, pad_width, mode="reflect")
            yield y, x, tile


def tile_image(
    image_path: str | Path,
    output_dir: str | Path,
    spec: TileSpec,
    *,
    prefix: str | None = None,
    save_format: str = "png",
) -> list[Path]:
    """Tile an on-disk image and save tiles to ``output_dir``.

    Returns the list of written tile paths.
    """

    src = Path(image_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = prefix or src.stem

    with Image.open(src) as im:
        arr = np.asarray(im.convert("RGB"))

    written: list[Path] = []
    for y, x, tile in iter_tiles(arr, spec):
        tile_img = Image.fromarray(tile.astype(np.uint8))
        name = f"{stem}_y{y:05d}_x{x:05d}.{save_format}"
        path = out_dir / name
        tile_img.save(path)
        written.append(path)
    return written
