"""CLI: fine-tune Stable Diffusion with LoRA on the disaster dataset."""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

from cv_diffusion.cli._shared import common_parser, init_logging, load_yaml_if_given
from cv_diffusion.models.lora import LoRAConfig
from cv_diffusion.training.train_lora import TrainLoRAConfig, train_lora
from cv_diffusion.utils.config import merge
from cv_diffusion.utils.logging import get_logger

logger = get_logger(__name__)


def _build_parser():
    parser = common_parser("LoRA fine-tune Stable Diffusion on disaster satellite tiles.")
    parser.add_argument("--train-data-dir", type=Path, default=None)
    parser.add_argument("--train-manifest-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--pretrained-model-id", type=str, default=None)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--alpha", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser


def _apply_overrides(cfg_dict: dict, args) -> TrainLoRAConfig:
    base = asdict(TrainLoRAConfig())
    merged = merge(base, cfg_dict)
    cli_overrides: dict = {}
    flat_cli = {
        "train_data_dir": args.train_data_dir,
        "train_manifest_csv": args.train_manifest_csv,
        "output_dir": args.output_dir,
        "pretrained_model_id": args.pretrained_model_id,
        "resolution": args.resolution,
        "learning_rate": args.learning_rate,
        "max_train_steps": args.max_train_steps,
        "train_batch_size": args.train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "mixed_precision": args.mixed_precision,
        "seed": args.seed,
    }
    for key, value in flat_cli.items():
        if value is None:
            continue
        if isinstance(value, Path):
            value = str(value)
        cli_overrides[key] = value
    merged = merge(merged, cli_overrides)

    lora_dict = merged.get("lora", {}) or {}
    if args.rank is not None:
        lora_dict["rank"] = args.rank
    if args.alpha is not None:
        lora_dict["alpha"] = args.alpha

    lora_cfg = LoRAConfig(**{k: v for k, v in lora_dict.items() if k in LoRAConfig.__dataclass_fields__})

    cleaned = {k: v for k, v in merged.items() if k != "lora" and k in TrainLoRAConfig.__dataclass_fields__}
    return TrainLoRAConfig(**cleaned, lora=lora_cfg)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    init_logging(args)
    cfg_dict = load_yaml_if_given(args)
    config = _apply_overrides(cfg_dict, args)
    if not config.train_data_dir and not config.train_manifest_csv:
        logger.error("Provide --train-data-dir or --train-manifest-csv (or set them in --config).")
        return 2
    train_lora(config)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
