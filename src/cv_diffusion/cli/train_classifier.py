"""CLI: train the downstream disaster classifier (single arm)."""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

from cv_diffusion.cli._shared import common_parser, init_logging, load_yaml_if_given
from cv_diffusion.training.train_classifier import TrainClassifierConfig, train_classifier
from cv_diffusion.utils.config import merge
from cv_diffusion.utils.logging import get_logger

logger = get_logger(__name__)


def _build_parser():
    parser = common_parser("Train the downstream disaster classifier.")
    parser.add_argument("--train-root", type=Path, default=None)
    parser.add_argument("--val-root", type=Path, default=None)
    parser.add_argument("--test-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser


def _apply_overrides(cfg_dict: dict, args) -> TrainClassifierConfig:
    base = asdict(TrainClassifierConfig(train_root="placeholder"))
    merged = merge(base, cfg_dict.get("classifier", cfg_dict))
    cli = {
        "train_root": args.train_root,
        "val_root": args.val_root,
        "test_root": args.test_root,
        "output_dir": args.output_dir,
        "backbone": args.backbone,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "image_size": args.image_size,
        "seed": args.seed,
    }
    overrides = {}
    for k, v in cli.items():
        if v is None:
            continue
        overrides[k] = str(v) if isinstance(v, Path) else v
    merged = merge(merged, overrides)
    keep = {k: v for k, v in merged.items() if k in TrainClassifierConfig.__dataclass_fields__}
    if not keep.get("train_root"):
        raise SystemExit("Provide --train-root (or set classifier.train_root in --config).")
    return TrainClassifierConfig(**keep)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    init_logging(args)
    cfg_dict = load_yaml_if_given(args)
    config = _apply_overrides(cfg_dict, args)
    train_classifier(config)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
