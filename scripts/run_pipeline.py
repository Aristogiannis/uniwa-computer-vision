#!/usr/bin/env python3
"""End-to-end pipeline driver.

Runs the four stages of the project in sequence, each behind a flag so you
can re-run individual steps as you iterate:

    1. preprocess     — tile + normalize raw scenes
    2. train-lora     — fine-tune SD with LoRA
    3. generate       — sample synthetic images
    4. evaluate       — FID + SSIM + 3-arm downstream protocol

Typical use::

    python scripts/run_pipeline.py --all --config configs/lora_sd15.yaml \\
        --real-train-root data/processed/xbd_classifier/train \\
        --real-test-root  data/processed/xbd_classifier/test

For an academic project most users will run each stage separately so they
can inspect intermediate outputs. This driver exists as a smoke test and to
make the README's example command actually executable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cv_diffusion.cli import evaluate as cli_evaluate
from cv_diffusion.cli import generate as cli_generate
from cv_diffusion.cli import preprocess as cli_preprocess
from cv_diffusion.cli import train_lora as cli_train_lora
from cv_diffusion.utils.logging import get_logger, setup_logging

logger = get_logger("cv_diffusion.pipeline")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true", help="Run every stage in order.")
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--train-lora", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--evaluate", action="store_true")

    parser.add_argument("--preprocess-config", type=Path, default=Path("configs/preprocess.yaml"))
    parser.add_argument("--lora-config", type=Path, default=Path("configs/lora_sd15.yaml"))
    parser.add_argument("--generation-config", type=Path, default=Path("configs/generation.yaml"))

    parser.add_argument("--real-train-root", type=Path, default=None)
    parser.add_argument("--real-val-root", type=Path, default=None)
    parser.add_argument("--real-test-root", type=Path, default=None)
    parser.add_argument("--synthetic-root", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--downstream-output", type=Path, default=Path("outputs/downstream"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    setup_logging("INFO")

    do = {
        "preprocess": args.all or args.preprocess,
        "train_lora": args.all or args.train_lora,
        "generate": args.all or args.generate,
        "evaluate": args.all or args.evaluate,
    }

    if do["preprocess"]:
        logger.info("=== Stage 1: preprocess ===")
        rc = cli_preprocess.main(["--config", str(args.preprocess_config)])
        if rc != 0:
            return rc

    if do["train_lora"]:
        logger.info("=== Stage 2: LoRA fine-tune ===")
        rc = cli_train_lora.main(["--config", str(args.lora_config)])
        if rc != 0:
            return rc

    if do["generate"]:
        logger.info("=== Stage 3: synthetic image generation ===")
        rc = cli_generate.main(["--config", str(args.generation_config)])
        if rc != 0:
            return rc

    if do["evaluate"]:
        logger.info("=== Stage 4: evaluation ===")
        if not (args.real_train_root and args.real_test_root):
            logger.error("--real-train-root and --real-test-root are required for --evaluate.")
            return 2
        rc = cli_evaluate.main([
            "downstream",
            "--real-train-root", str(args.real_train_root),
            "--real-test-root", str(args.real_test_root),
            *(['--real-val-root', str(args.real_val_root)] if args.real_val_root else []),
            "--synthetic-root", str(args.synthetic_root),
            "--output-root", str(args.downstream_output),
        ])
        if rc != 0:
            return rc

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
