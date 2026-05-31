import random

import numpy as np
import pytest

from cv_diffusion.utils.io import IMAGE_EXTENSIONS, ensure_dir, list_images, save_image_grid
from cv_diffusion.utils.seed import seed_everything


def test_seed_everything_is_deterministic():
    seed_everything(123)
    a = [random.random() for _ in range(5)] + np.random.rand(5).tolist()
    seed_everything(123)
    b = [random.random() for _ in range(5)] + np.random.rand(5).tolist()
    assert a == b


def test_list_images_returns_sorted(synthetic_dataset_dir):
    images = list_images(synthetic_dataset_dir)
    assert len(images) == 8
    assert images == sorted(images)
    assert all(p.suffix.lower() in IMAGE_EXTENSIONS for p in images)


def test_save_image_grid(tmp_path, synthetic_dataset_dir):
    from PIL import Image

    images = [Image.open(p).convert("RGB").resize((32, 32)) for p in list_images(synthetic_dataset_dir)]
    out = save_image_grid(images, tmp_path / "grid.png", cols=4)
    assert out.exists()


def test_ensure_dir_creates_nested(tmp_path):
    new = tmp_path / "a" / "b" / "c"
    result = ensure_dir(new)
    assert result.exists() and result.is_dir()
