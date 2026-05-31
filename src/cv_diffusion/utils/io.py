"""IO helpers for images and result artifacts."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_images(root: str | Path, *, recursive: bool = True) -> list[Path]:
    """Return all image files under ``root`` with stable lexicographic order."""

    base = Path(root)
    if not base.exists():
        return []
    iterator: Iterable[Path] = base.rglob("*") if recursive else base.glob("*")
    out = [p for p in iterator if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(out)


def save_image_grid(
    images: Sequence[Image.Image],
    output_path: str | Path,
    *,
    cols: int | None = None,
    padding: int = 4,
    background: tuple[int, int, int] = (255, 255, 255),
) -> Path:
    """Tile PIL images into a single grid PNG. Used for qualitative reports."""

    if not images:
        raise ValueError("save_image_grid() requires at least one image.")

    n = len(images)
    cols = cols or int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    w, h = images[0].size

    grid = Image.new("RGB", (cols * w + (cols + 1) * padding, rows * h + (rows + 1) * padding), background)
    for idx, img in enumerate(images):
        r, c = divmod(idx, cols)
        x = padding + c * (w + padding)
        y = padding + r * (h + padding)
        grid.paste(img.convert("RGB"), (x, y))

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out)
    return out
