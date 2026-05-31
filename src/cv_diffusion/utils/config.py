"""YAML-based configuration loading with light schema validation.

We deliberately avoid heavyweight config libraries (Hydra/OmegaConf). YAML +
dataclasses keeps the project easy to read for first-time contributors while
still giving us type-checked entry points.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file as a plain dict."""

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Top-level YAML in {p} must be a mapping, got {type(data).__name__}")
    return data


def save_config(config: Any, path: str | Path) -> None:
    """Write a config (dict or dataclass) to YAML."""

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(config) if is_dataclass(config) else dict(config)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def merge(base: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    """Recursive dict merge — ``override`` wins. Used for CLI argument overlays."""

    if not override:
        return dict(base)
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = v
    return out
