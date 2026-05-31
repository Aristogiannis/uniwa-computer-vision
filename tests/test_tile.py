import numpy as np
import pytest

from cv_diffusion.preprocessing.tile import TileSpec, iter_tiles


def test_tile_counts_exact_grid():
    img = np.zeros((1024, 1024, 3), dtype=np.uint8)
    tiles = list(iter_tiles(img, TileSpec(size=512, stride=512)))
    assert len(tiles) == 4  # 2x2 grid
    for _, _, t in tiles:
        assert t.shape == (512, 512, 3)


def test_tile_drop_partial_skips_edges():
    img = np.zeros((600, 600, 3), dtype=np.uint8)
    tiles = list(iter_tiles(img, TileSpec(size=512, stride=512, drop_partial=True)))
    assert len(tiles) == 1


def test_tile_no_drop_pads_to_full_size():
    img = np.zeros((600, 600, 3), dtype=np.uint8)
    tiles = list(iter_tiles(img, TileSpec(size=512, stride=512, drop_partial=False)))
    assert len(tiles) >= 1
    for _, _, t in tiles:
        assert t.shape == (512, 512, 3)


def test_tile_overlap():
    img = np.zeros((1024, 1024, 3), dtype=np.uint8)
    tiles = list(iter_tiles(img, TileSpec(size=512, stride=256)))
    # Origins along one axis: 0, 256, 512 -> 3, so 3x3 = 9.
    assert len(tiles) == 9


def test_tile_rejects_invalid_dim():
    with pytest.raises(ValueError):
        list(iter_tiles(np.zeros((4,)), TileSpec(size=2, stride=2)))
