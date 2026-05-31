"""CLI: evaluate generated images (FID, SSIM) and run the 3-arm protocol."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

from cv_diffusion.cli._shared import common_parser, init_logging
from cv_diffusion.evaluation.downstream import (
    DownstreamProtocolConfig,
    run_downstream_protocol,
)
from cv_diffusion.evaluation.fid import compute_fid
from cv_diffusion.evaluation.ssim import compute_set_ssim
from cv_diffusion.utils.logging import get_logger

logger = get_logger(__name__)


def _build_parser():
    parser = common_parser("Evaluate synthetic dataset and run the downstream protocol.")
    sub = parser.add_subparsers(dest="command", required=True)

    fid_p = sub.add_parser("fid", help="Compute FID between two folders.")
    fid_p.add_argument("--real-dir", type=Path, required=True)
    fid_p.add_argument("--fake-dir", type=Path, required=True)
    fid_p.add_argument("--mode", choices=["clean", "legacy_pytorch", "legacy_tensorflow"], default="clean")
    fid_p.add_argument("--out", type=Path, default=None, help="Optional JSON file for the result.")

    ssim_p = sub.add_parser("ssim", help="Compute set-level SSIM (mean + nearest).")
    ssim_p.add_argument("--real-dir", type=Path, required=True)
    ssim_p.add_argument("--fake-dir", type=Path, required=True)
    ssim_p.add_argument("--image-size", type=int, default=256)
    ssim_p.add_argument("--max-real", type=int, default=256)
    ssim_p.add_argument("--max-fake", type=int, default=256)
    ssim_p.add_argument("--out", type=Path, default=None)

    downstream_p = sub.add_parser("downstream", help="Run the 3-arm downstream protocol.")
    downstream_p.add_argument("--real-train-root", type=Path, required=True)
    downstream_p.add_argument("--real-val-root", type=Path, default=None)
    downstream_p.add_argument("--real-test-root", type=Path, required=True)
    downstream_p.add_argument("--synthetic-root", type=Path, default=None)
    downstream_p.add_argument("--output-root", type=Path, default=Path("outputs/downstream"))
    downstream_p.add_argument("--backbone", type=str, default="resnet18")
    downstream_p.add_argument("--epochs", type=int, default=20)
    downstream_p.add_argument("--batch-size", type=int, default=32)
    downstream_p.add_argument("--learning-rate", type=float, default=3e-4)
    downstream_p.add_argument("--seed", type=int, default=42)
    downstream_p.add_argument("--skip-synthetic-only", action="store_true")
    return parser


def _save_or_print(result: dict, out: Path | None) -> None:
    payload = json.dumps(result, indent=2)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload)
    print(payload)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    init_logging(args)

    if args.command == "fid":
        score = compute_fid(args.real_dir, args.fake_dir, mode=args.mode)
        _save_or_print({"fid": score, "mode": args.mode}, args.out)
        return 0

    if args.command == "ssim":
        result = compute_set_ssim(
            args.real_dir,
            args.fake_dir,
            image_size=args.image_size,
            max_real=args.max_real,
            max_fake=args.max_fake,
        )
        _save_or_print(result, args.out)
        return 0

    if args.command == "downstream":
        cfg = DownstreamProtocolConfig(
            real_train_root=str(args.real_train_root),
            real_val_root=str(args.real_val_root) if args.real_val_root else None,
            real_test_root=str(args.real_test_root),
            synthetic_root=str(args.synthetic_root) if args.synthetic_root else None,
            output_root=str(args.output_root),
            backbone=args.backbone,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            run_synthetic_only=not args.skip_synthetic_only,
        )
        result = run_downstream_protocol(cfg)
        print(json.dumps(asdict(result), indent=2))
        return 0

    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
