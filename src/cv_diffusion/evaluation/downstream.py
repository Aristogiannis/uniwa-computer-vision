"""Downstream evaluation: does synthetic augmentation help the classifier?

Implements the canonical 3-arm protocol used by Frid-Adar et al. (2018) and
the broader synthetic-augmentation literature:

1. **Real-only** — classifier trained on real disaster tiles + classical aug
2. **Real + synthetic** — same real data plus N synthetic tiles per class
3. **Synthetic-only** — sanity-check arm

All arms are evaluated on a *real-only* test set with fixed seed; we report
accuracy, macro-F1 and per-class recall, plus the delta between arms 1 and 2.

This module is the orchestrator — actual training calls into
:func:`cv_diffusion.training.train_classifier.train_classifier` and the
arms differ only in their ``train_root``.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from cv_diffusion.training.train_classifier import (
    TrainClassifierConfig,
    train_classifier,
)
from cv_diffusion.utils.io import ensure_dir, list_images
from cv_diffusion.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DownstreamProtocolConfig:
    real_train_root: str
    real_val_root: Optional[str]
    real_test_root: str
    synthetic_root: Optional[str]
    output_root: str = "outputs/downstream"
    backbone: str = "resnet18"
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 3e-4
    seed: int = 42
    run_synthetic_only: bool = True


@dataclass
class DownstreamProtocolResult:
    real_only: dict
    real_plus_synth: Optional[dict] = None
    synth_only: Optional[dict] = None
    delta_accuracy: Optional[float] = None
    delta_macro_f1: Optional[float] = None
    config: dict = field(default_factory=dict)


def _build_combined_dir(real_root: str | Path, synth_root: str | Path, dest: Path) -> Path:
    """Symlink (or copy) real + synthetic folders into a single train root."""

    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    real_root = Path(real_root)
    synth_root = Path(synth_root)
    real_classes = {p.name for p in real_root.iterdir() if p.is_dir()}
    synth_classes = {p.name for p in synth_root.iterdir() if p.is_dir()}
    classes = sorted(real_classes | synth_classes)

    for cls in classes:
        cls_dir = dest / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for src_root, tag in ((real_root, "real"), (synth_root, "synth")):
            src = src_root / cls
            if not src.exists():
                continue
            for i, img in enumerate(list_images(src, recursive=True)):
                link = cls_dir / f"{tag}_{i:06d}{img.suffix.lower()}"
                try:
                    link.symlink_to(img.resolve())
                except (OSError, NotImplementedError):
                    shutil.copy2(img, link)
    return dest


def _arm_metrics(metrics: dict) -> dict:
    out = {
        "best_val_acc": metrics.get("best_val_acc"),
    }
    if "test" in metrics:
        out["test_accuracy"] = metrics["test"]["accuracy"]
        out["test_macro_f1"] = metrics["test"]["macro_f1"]
        out["test_weighted_f1"] = metrics["test"]["weighted_f1"]
        out["per_class"] = {
            cls: metrics["test"]["report"].get(cls)
            for cls in metrics["test"]["classes"]
        }
    return out


def run_downstream_protocol(config: DownstreamProtocolConfig) -> DownstreamProtocolResult:
    output_root = ensure_dir(config.output_root)
    (output_root / "protocol_config.json").write_text(json.dumps(asdict(config), indent=2))

    logger.info("=== Arm 1: real-only ===")
    real_only = train_classifier(
        TrainClassifierConfig(
            train_root=config.real_train_root,
            val_root=config.real_val_root,
            test_root=config.real_test_root,
            output_dir=str(output_root / "arm_real_only"),
            backbone=config.backbone,
            epochs=config.epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            seed=config.seed,
        )
    )

    real_plus_synth = None
    delta_acc = None
    delta_f1 = None
    if config.synthetic_root is not None:
        combined_dir = _build_combined_dir(
            config.real_train_root, config.synthetic_root, output_root / "combined_train"
        )
        logger.info("=== Arm 2: real + synthetic ===")
        real_plus_synth = train_classifier(
            TrainClassifierConfig(
                train_root=str(combined_dir),
                val_root=config.real_val_root,
                test_root=config.real_test_root,
                output_dir=str(output_root / "arm_real_plus_synth"),
                backbone=config.backbone,
                epochs=config.epochs,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                seed=config.seed,
            )
        )
        if "test" in real_only and "test" in real_plus_synth:
            delta_acc = real_plus_synth["test"]["accuracy"] - real_only["test"]["accuracy"]
            delta_f1 = real_plus_synth["test"]["macro_f1"] - real_only["test"]["macro_f1"]

    synth_only = None
    if config.run_synthetic_only and config.synthetic_root is not None:
        logger.info("=== Arm 3: synthetic-only (sanity check) ===")
        synth_only = train_classifier(
            TrainClassifierConfig(
                train_root=config.synthetic_root,
                val_root=config.real_val_root,
                test_root=config.real_test_root,
                output_dir=str(output_root / "arm_synth_only"),
                backbone=config.backbone,
                epochs=config.epochs,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                seed=config.seed,
            )
        )

    result = DownstreamProtocolResult(
        real_only=_arm_metrics(real_only),
        real_plus_synth=_arm_metrics(real_plus_synth) if real_plus_synth is not None else None,
        synth_only=_arm_metrics(synth_only) if synth_only is not None else None,
        delta_accuracy=delta_acc,
        delta_macro_f1=delta_f1,
        config=asdict(config),
    )

    (output_root / "summary.json").write_text(json.dumps(asdict(result), indent=2))
    logger.info("Protocol complete: delta_acc=%s delta_f1=%s", delta_acc, delta_f1)
    return result
