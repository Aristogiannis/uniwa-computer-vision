import numpy as np
import pytest

from cv_diffusion.preprocessing.spectral import (
    RGB_BAND_ORDER,
    SWIR_BAND_ORDER,
    rgb_from_sentinel2,
    select_bands,
)


def test_select_bands_from_hwc():
    stack = np.stack([np.full((8, 8), i, dtype=np.float32) for i in range(13)], axis=-1)
    rgb = select_bands(stack, RGB_BAND_ORDER)
    assert rgb.shape == (8, 8, 3)
    # B04, B03, B02 correspond to indices 3, 2, 1.
    assert rgb[..., 0].mean() == 3
    assert rgb[..., 1].mean() == 2
    assert rgb[..., 2].mean() == 1


def test_select_bands_accepts_chw_layout():
    stack = np.stack([np.full((8, 8), i, dtype=np.float32) for i in range(13)], axis=0)
    swir = select_bands(stack, SWIR_BAND_ORDER)
    assert swir.shape == (8, 8, 3)
    assert swir[..., 0].mean() == 12  # B12


def test_rgb_from_sentinel2_dispatches_correctly():
    stack = np.zeros((4, 4, 13), dtype=np.float32)
    assert rgb_from_sentinel2(stack, "swir").shape == (4, 4, 3)
    with pytest.raises(ValueError):
        rgb_from_sentinel2(stack, "no-such-composition")
