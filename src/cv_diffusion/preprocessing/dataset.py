"""PyTorch datasets for the project.

Two datasets cover our needs:

* :class:`SatelliteFolderDataset` — a generic folder-of-images dataset with
  optional class subfolders (``ImageFolder``-style) used both for the
  classifier and as the LoRA-training image source.
* :class:`DisasterImageDataset` — pairs each image with a natural-language
  caption derived from its category, the format expected by the
  ``train_text_to_image_lora`` recipe from the diffusers library.

A small CSV manifest format is also supported so that captions can be
human-edited without renaming files.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from PIL import Image
from torch.utils.data import Dataset

from cv_diffusion.preprocessing.prompts import (
    DISASTER_PROMPT_TEMPLATES,
    build_text_prompt,
)
from cv_diffusion.utils.io import IMAGE_EXTENSIONS, list_images


__all__ = [
    "DISASTER_PROMPT_TEMPLATES",
    "DisasterImageDataset",
    "SatelliteFolderDataset",
    "build_text_prompt",
]


@dataclass
class _Sample:
    path: Path
    label: int
    caption: str
    category: str


class SatelliteFolderDataset(Dataset):
    """``ImageFolder``-style dataset for the disaster classifier and LoRA training.

    Expects ``root/<class_name>/*.png`` (or any supported image extension).
    The list of classes is sorted alphabetically for deterministic label
    indices across runs.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        transform: Optional[Callable] = None,
        classes: Optional[list[str]] = None,
        include_synthetic: bool = True,
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

        subdirs = [p for p in self.root.iterdir() if p.is_dir()]
        if not subdirs:
            raise ValueError(
                f"No class subdirectories found in {self.root}. Expected "
                "ImageFolder layout."
            )

        if classes is None:
            classes = sorted(p.name for p in subdirs)
        self.classes: list[str] = list(classes)
        self.class_to_idx: dict[str, int] = {c: i for i, c in enumerate(self.classes)}

        self.samples: list[_Sample] = []
        for cls in self.classes:
            cls_dir = self.root / cls
            if not cls_dir.exists():
                continue
            for img_path in list_images(cls_dir, recursive=True):
                if not include_synthetic and "synthetic" in img_path.parts:
                    continue
                self.samples.append(
                    _Sample(
                        path=img_path,
                        label=self.class_to_idx[cls],
                        caption=build_text_prompt(cls),
                        category=cls,
                    )
                )

        if not self.samples:
            raise ValueError(f"No images found under {self.root}")

        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with Image.open(sample.path) as im:
            image = im.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, sample.label


class DisasterImageDataset(Dataset):
    """Image + caption dataset used to fine-tune Stable Diffusion via LoRA.

    Two sources are supported:

    1. A class-folder layout (``root/wildfire/*.png``); captions are generated
       by :func:`build_text_prompt`.
    2. A CSV manifest with columns ``image_path,caption,category`` — useful
       when curating captions by hand.
    """

    def __init__(
        self,
        root: Optional[str | Path] = None,
        *,
        manifest_csv: Optional[str | Path] = None,
        transform: Optional[Callable] = None,
        image_column: str = "image_path",
        caption_column: str = "caption",
        category_column: str = "category",
    ) -> None:
        if root is None and manifest_csv is None:
            raise ValueError("Either root or manifest_csv must be provided.")

        self.transform = transform
        self.samples: list[_Sample] = []

        if manifest_csv is not None:
            manifest_path = Path(manifest_csv)
            with manifest_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                base = manifest_path.parent
                for row in reader:
                    rel = Path(row[image_column])
                    image_path = rel if rel.is_absolute() else (base / rel).resolve()
                    if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                        continue
                    category = row.get(category_column, "unknown")
                    caption = row.get(caption_column) or build_text_prompt(category)
                    self.samples.append(
                        _Sample(path=image_path, label=0, caption=caption, category=category)
                    )

        if root is not None:
            root_path = Path(root)
            subdirs = [p for p in root_path.iterdir() if p.is_dir()]
            if subdirs:
                for cls_dir in sorted(subdirs):
                    cls = cls_dir.name
                    for i, img_path in enumerate(list_images(cls_dir, recursive=True)):
                        self.samples.append(
                            _Sample(
                                path=img_path,
                                label=0,
                                caption=build_text_prompt(cls, index=i),
                                category=cls,
                            )
                        )
            else:
                # Flat folder fallback — single "unknown" category.
                for img_path in list_images(root_path, recursive=True):
                    self.samples.append(
                        _Sample(
                            path=img_path,
                            label=0,
                            caption=build_text_prompt("unknown"),
                            category="unknown",
                        )
                    )

        if not self.samples:
            raise ValueError("DisasterImageDataset is empty — check root/manifest paths.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        with Image.open(sample.path) as im:
            image = im.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return {
            "pixel_values": image,
            "caption": sample.caption,
            "category": sample.category,
        }
