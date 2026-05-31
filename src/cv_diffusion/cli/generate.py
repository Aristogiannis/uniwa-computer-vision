"""CLI: generate a synthetic disaster dataset using the fine-tuned LoRA."""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

from cv_diffusion.cli._shared import common_parser, init_logging, load_yaml_if_given
from cv_diffusion.generation.generate import (
    GenerationJob,
    GenerationRunConfig,
    expand_default_jobs,
    generate_synthetic_dataset,
)
from cv_diffusion.utils.config import merge
from cv_diffusion.utils.logging import get_logger

logger = get_logger(__name__)


def _build_parser():
    parser = common_parser("Generate synthetic disaster satellite images.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--lora-weights", type=str, default=None,
                        help="Path or HF repo id with the LoRA adapter to load.")
    parser.add_argument("--lora-scale", type=float, default=None)
    parser.add_argument("--base-model-id", type=str, default=None)
    parser.add_argument("--per-class", type=int, default=None,
                        help="Number of synthetic images per category (uses the default category set).")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--scheduler", type=str, default=None)
    parser.add_argument("--torch-dtype", choices=["float16", "bfloat16", "float32"], default=None)
    parser.add_argument("--category", action="append", default=None,
                        help="Restrict generation to these categories (repeatable). Default: all known.")
    return parser


def _apply_overrides(cfg_dict: dict, args) -> tuple[GenerationRunConfig, list[GenerationJob]]:
    base = asdict(GenerationRunConfig())
    merged = merge(base, cfg_dict.get("generation", cfg_dict))

    cli_overrides: dict = {}
    mapping = {
        "output_dir": args.output_dir,
        "lora_weights": args.lora_weights,
        "lora_scale": args.lora_scale,
        "base_model_id": args.base_model_id,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "height": args.height,
        "width": args.width,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "scheduler": args.scheduler,
        "torch_dtype": args.torch_dtype,
    }
    for key, value in mapping.items():
        if value is None:
            continue
        if isinstance(value, Path):
            value = str(value)
        cli_overrides[key] = value
    merged = merge(merged, cli_overrides)

    keep = {k: v for k, v in merged.items() if k in GenerationRunConfig.__dataclass_fields__}
    run_cfg = GenerationRunConfig(**keep)

    jobs_data = cfg_dict.get("jobs") or []
    per_class = args.per_class if args.per_class is not None else 100
    if jobs_data:
        jobs = [GenerationJob(**j) for j in jobs_data]
    else:
        jobs = expand_default_jobs(per_class=per_class)

    if args.category:
        wanted = set(args.category)
        jobs = [j for j in jobs if j.category in wanted]
        if not jobs:
            raise SystemExit(f"No matching jobs for categories: {sorted(wanted)}")

    return run_cfg, jobs


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    init_logging(args)
    cfg_dict = load_yaml_if_given(args)
    run_cfg, jobs = _apply_overrides(cfg_dict, args)
    generate_synthetic_dataset(jobs, run_cfg)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
