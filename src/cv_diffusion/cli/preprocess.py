"""CLI: tile and normalize a folder of raw satellite scenes."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

from cv_diffusion.cli._shared import common_parser, init_logging
from cv_diffusion.preprocessing.normalize import percentile_normalize
from cv_diffusion.preprocessing.tile import TileSpec, iter_tiles
from cv_diffusion.utils.io import ensure_dir, list_images
from cv_diffusion.utils.logging import get_logger

logger = get_logger(__name__)


def _build_parser():
    parser = common_parser("Preprocess satellite images: percentile-normalize + tile.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Folder of raw images.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Where to write tiles.")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--drop-partial", action="store_true", default=True)
    parser.add_argument("--no-drop-partial", dest="drop_partial", action="store_false")
    parser.add_argument("--percentile-lower", type=float, default=2.0)
    parser.add_argument("--percentile-upper", type=float, default=98.0)
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=True,
        help="Recurse into subdirectories (the default).",
    )
    parser.add_argument("--no-recursive", dest="recursive", action="store_false")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    init_logging(args)

    images = list_images(args.input_dir, recursive=args.recursive)
    if not images:
        logger.error("No images found under %s", args.input_dir)
        return 2

    spec = TileSpec(size=args.tile_size, stride=args.stride, drop_partial=args.drop_partial)
    out_dir = ensure_dir(args.output_dir)
    total_tiles = 0

    for img_path in images:
        with Image.open(img_path) as im:
            arr = np.asarray(im.convert("RGB"))
        arr = (percentile_normalize(arr, lower=args.percentile_lower, upper=args.percentile_upper) * 255.0).astype(np.uint8)

        class_dir = out_dir / img_path.parent.name if img_path.parent != args.input_dir else out_dir
        class_dir.mkdir(parents=True, exist_ok=True)

        n = 0
        for y, x, tile in iter_tiles(arr, spec):
            out_name = f"{img_path.stem}_y{y:05d}_x{x:05d}.png"
            Image.fromarray(tile).save(class_dir / out_name)
            n += 1
        logger.info("%s -> %d tiles in %s", img_path.name, n, class_dir)
        total_tiles += n

    logger.info("Preprocessing complete: %d tiles in %s", total_tiles, out_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
