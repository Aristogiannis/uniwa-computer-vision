"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.fixture()
def synthetic_dataset_dir(tmp_path: Path) -> Path:
    """Create a small fake dataset with two classes and 4 images each."""

    rng = np.random.default_rng(0)
    for cls in ("flood", "wildfire"):
        cls_dir = tmp_path / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(4):
            arr = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
            Image.fromarray(arr).save(cls_dir / f"{cls}_{i:04d}.png")
    return tmp_path
