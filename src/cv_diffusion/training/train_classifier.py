"""Train (and re-train) the downstream disaster classifier.

This is the workhorse of the *downstream evaluation* phase of the project.
The expected protocol — borrowed from Frid-Adar et al. (2018) and the
synthetic-augmentation literature — is to compare three arms:

1. **Real-only**: classifier trained on real disaster tiles + classical aug.
2. **Real + synthetic**: same real data plus N synthetic tiles per class
   produced by the LoRA-fine-tuned diffusion model.
3. **Synthetic-only** (sanity check).

The CLI script ``cv-train-classifier`` lets the user point this function at
any of those three configurations by passing different ``train_root``
directories.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from cv_diffusion.utils.logging import get_logger
from cv_diffusion.utils.seed import seed_everything

logger = get_logger(__name__)


@dataclass
class TrainClassifierConfig:
    train_root: str
    val_root: Optional[str] = None
    test_root: Optional[str] = None
    output_dir: str = "outputs/classifier"
    backbone: str = "resnet18"
    pretrained: bool = True
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    epochs: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    early_stop_patience: int = 5
    seed: int = 42
    mixed_precision: bool = True
    class_weights: bool = True


def _build_dataloaders(config: TrainClassifierConfig):
    import torch
    from torch.utils.data import DataLoader

    from cv_diffusion.models.classifier import default_transforms
    from cv_diffusion.preprocessing.dataset import SatelliteFolderDataset

    train_set = SatelliteFolderDataset(
        config.train_root, transform=default_transforms(config.image_size, train=True)
    )
    classes = train_set.classes

    val_set = None
    if config.val_root:
        val_set = SatelliteFolderDataset(
            config.val_root,
            transform=default_transforms(config.image_size, train=False),
            classes=classes,
            include_synthetic=False,
        )
    test_set = None
    if config.test_root:
        test_set = SatelliteFolderDataset(
            config.test_root,
            transform=default_transforms(config.image_size, train=False),
            classes=classes,
            include_synthetic=False,
        )

    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = (
        DataLoader(val_set, batch_size=config.batch_size, num_workers=config.num_workers)
        if val_set is not None
        else None
    )
    test_loader = (
        DataLoader(test_set, batch_size=config.batch_size, num_workers=config.num_workers)
        if test_set is not None
        else None
    )

    return train_loader, val_loader, test_loader, classes


def _accuracy(logits, labels) -> float:
    import torch

    preds = torch.argmax(logits, dim=1)
    return float((preds == labels).float().mean().item())


def train_classifier(config: TrainClassifierConfig) -> dict:
    """Train the disaster classifier and return a metrics dict."""

    import torch
    from torch import nn

    from cv_diffusion.models.classifier import build_classifier, class_weights_from_counts

    seed_everything(config.seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader, test_loader, classes = _build_dataloaders(config)
    logger.info("Detected classes (%d): %s", len(classes), classes)

    counts = [0] * len(classes)
    for sample in train_loader.dataset.samples:
        counts[sample.label] += 1
    logger.info("Class counts: %s", dict(zip(classes, counts)))

    model = build_classifier(num_classes=len(classes), backbone=config.backbone, pretrained=config.pretrained)
    model.to(device)

    if config.class_weights and any(c > 0 for c in counts):
        weights = class_weights_from_counts(counts).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=config.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=config.mixed_precision and device == "cuda")

    best_val_acc = -1.0
    epochs_without_improvement = 0
    history: list[dict] = []
    best_state = None

    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=config.mixed_precision and device == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item())
            running_acc += _accuracy(logits, labels)
        scheduler.step()
        train_loss = running_loss / max(len(train_loader), 1)
        train_acc = running_acc / max(len(train_loader), 1)

        val_acc = None
        if val_loader is not None:
            val_acc = _evaluate(model, val_loader, device)
        record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(record)
        logger.info("[ep %d/%d] loss=%.4f acc=%.4f val_acc=%s",
                    epoch + 1, config.epochs, train_loss, train_acc,
                    f"{val_acc:.4f}" if val_acc is not None else "n/a")

        improved = val_acc is None or val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc if val_acc is not None else train_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stop_patience:
                logger.info("Early stopping triggered.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    ckpt_path = output_dir / "classifier.pt"
    torch.save({"state_dict": model.state_dict(), "classes": classes, "config": asdict(config)}, ckpt_path)
    (output_dir / "history.json").write_text(json.dumps(history, indent=2))

    metrics = {"classes": classes, "history": history, "best_val_acc": best_val_acc}

    if test_loader is not None:
        report = _detailed_evaluation(model, test_loader, device, classes)
        metrics["test"] = report
        (output_dir / "test_report.json").write_text(json.dumps(report, indent=2))
        logger.info("Test accuracy: %.4f | macro-F1: %.4f", report["accuracy"], report["macro_f1"])

    return metrics


def _evaluate(model, loader, device) -> float:
    import torch

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            preds = torch.argmax(model(images), dim=1)
            correct += int((preds == labels).sum().item())
            total += labels.size(0)
    return correct / max(total, 1)


def _detailed_evaluation(model, loader, device, classes: list[str]) -> dict:
    import numpy as np
    import torch
    from sklearn.metrics import classification_report, confusion_matrix

    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            preds = torch.argmax(model(images), dim=1).detach().cpu().numpy()
            y_true.extend(labels.numpy().tolist())
            y_pred.extend(preds.tolist())

    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    accuracy = float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))
    return {
        "accuracy": accuracy,
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "report": report,
        "confusion_matrix": cm.tolist(),
        "classes": classes,
    }
