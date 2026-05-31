from dataclasses import dataclass

import pytest

from cv_diffusion.utils.config import load_config, merge, save_config


@dataclass
class _Demo:
    a: int
    b: str


def test_round_trip(tmp_path):
    path = tmp_path / "c.yaml"
    save_config(_Demo(a=1, b="hi"), path)
    data = load_config(path)
    assert data == {"a": 1, "b": "hi"}


def test_merge_deeply():
    base = {"x": 1, "nested": {"a": 1, "b": 2}}
    override = {"nested": {"b": 99, "c": 7}, "y": 2}
    out = merge(base, override)
    assert out == {"x": 1, "nested": {"a": 1, "b": 99, "c": 7}, "y": 2}


def test_load_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")
