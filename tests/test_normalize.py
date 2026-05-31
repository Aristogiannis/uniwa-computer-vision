import numpy as np
import pytest

from cv_diffusion.preprocessing.normalize import (
    minmax_normalize,
    percentile_normalize,
    to_minus_one_one,
)


def test_percentile_normalize_outputs_unit_interval():
    rng = np.random.default_rng(123)
    arr = rng.uniform(50, 5000, size=(32, 32, 3)).astype(np.float32)
    arr[0, 0, 0] = 1e6  # outlier should not blow up the scale
    normed = percentile_normalize(arr)
    assert normed.shape == arr.shape
    assert normed.min() >= 0.0
    assert normed.max() <= 1.0
    # The 2-98 clip should pull the dynamic range close to unit.
    assert normed.max() > 0.95


def test_percentile_normalize_2d():
    arr = np.linspace(0, 1, 32 * 32).reshape(32, 32).astype(np.float32)
    normed = percentile_normalize(arr)
    assert normed.shape == arr.shape
    assert normed.min() >= 0.0
    assert normed.max() <= 1.0


def test_percentile_rejects_inverted_bounds():
    with pytest.raises(ValueError):
        percentile_normalize(np.zeros((4, 4), dtype=np.float32), lower=80.0, upper=20.0)


def test_minmax_and_signed_range():
    arr = np.arange(100, dtype=np.float32).reshape(10, 10)
    m = minmax_normalize(arr)
    s = to_minus_one_one(m)
    assert pytest.approx(m.min(), abs=1e-6) == 0.0
    assert pytest.approx(m.max(), abs=1e-6) == 1.0
    assert pytest.approx(s.min(), abs=1e-6) == -1.0
    assert pytest.approx(s.max(), abs=1e-6) == 1.0
