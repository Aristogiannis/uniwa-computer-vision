"""Disaster classifier used for the downstream evaluation experiment.

We default to a small ResNet-18 backbone from ``timm``: it's fast to train
end-to-end on a single GPU, well-studied on EuroSAT, and avoids the
classifier dominating training cost relative to the diffusion fine-tune.

For RGB satellite tiles the classifier sees the same channel statistics as
ImageNet (mean/std normalisation applied to ``[0, 1]`` floats). Multispectral
inputs should be reduced to a 3-band composite before being passed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn


@dataclass
class ClassifierConfig:
    backbone: str = "resnet18"
    num_classes: int = 3
    pretrained: bool = True
    drop_rate: float = 0.1
    image_size: int = 224
    in_channels: int = 3


class DisasterClassifier(nn.Module):
    """Thin wrapper around a ``timm`` backbone with a fresh classification head."""

    def __init__(self, config: ClassifierConfig) -> None:
        super().__init__()
        import timm

        self.config = config
        self.backbone = timm.create_model(
            config.backbone,
            pretrained=config.pretrained,
            num_classes=config.num_classes,
            drop_rate=config.drop_rate,
            in_chans=config.in_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(x), dim=-1)


def build_classifier(
    num_classes: int,
    *,
    backbone: str = "resnet18",
    pretrained: bool = True,
    drop_rate: float = 0.1,
) -> DisasterClassifier:
    """Convenience constructor used by the CLI."""

    return DisasterClassifier(
        ClassifierConfig(
            backbone=backbone,
            num_classes=num_classes,
            pretrained=pretrained,
            drop_rate=drop_rate,
        )
    )


def default_transforms(image_size: int = 224, train: bool = True):
    """Return a basic torchvision transform pipeline matching ImageNet stats."""

    from torchvision import transforms

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    if train:
        return transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def class_weights_from_counts(counts: Sequence[int]) -> torch.Tensor:
    """Inverse-frequency class weights for unbalanced disaster datasets."""

    counts_tensor = torch.tensor(counts, dtype=torch.float32)
    weights = counts_tensor.sum() / (len(counts) * counts_tensor.clamp(min=1.0))
    return weights
