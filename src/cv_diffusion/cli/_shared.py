"""Tiny helpers shared across the CLI entry points."""

from __future__ import annotations

import argparse
from pathlib import Path

from cv_diffusion.utils.config import load_config
from cv_diffusion.utils.logging import setup_logging


def common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML config file. CLI flags override config values.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional file to mirror log output to.",
    )
    return parser


def init_logging(args) -> None:
    setup_logging(level=args.log_level, log_file=args.log_file)


def load_yaml_if_given(args) -> dict:
    if getattr(args, "config", None) is None:
        return {}
    return load_config(args.config)
