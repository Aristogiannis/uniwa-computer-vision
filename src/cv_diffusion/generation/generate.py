"""Batch generation of synthetic disaster tiles using the fine-tuned LoRA.

The output directory mirrors the ``SatelliteFolderDataset`` layout so the
classifier can be retrained on it with no additional plumbing:

    <output_dir>/
        flood/
            flood_000000.png
            ...
        wildfire/
            wildfire_000000.png
            ...
        pre_disaster/
            ...

We also write a ``manifest.csv`` recording the prompt, seed and class for
each generated image — required for the project report and for downstream
filtering.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

from cv_diffusion.models.diffusion import (
    DiffusionConfig,
    GenerationParams,
    StableDiffusionWrapper,
)
from cv_diffusion.preprocessing.prompts import (
    DISASTER_PROMPT_TEMPLATES,
    build_text_prompt,
)
from cv_diffusion.utils.io import ensure_dir
from cv_diffusion.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_NEGATIVE_PROMPT = (
    "low resolution, blurry, oversaturated, cartoon, illustration, painting, "
    "people, faces, text, watermark, photoshopped, dramatic lighting"
)


@dataclass
class GenerationJob:
    """A single class to generate.

    Fields
    ------
    category:
        Folder name and key into the prompt template bank.
    num_images:
        How many synthetic samples to produce.
    prompts:
        Optional list of explicit prompts. If empty we cycle through the
        ``DISASTER_PROMPT_TEMPLATES`` bank for the category.
    """

    category: str
    num_images: int = 100
    prompts: list[str] = field(default_factory=list)


@dataclass
class GenerationRunConfig:
    output_dir: str = "data/synthetic"
    base_model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    lora_weights: str | None = None
    lora_scale: float = 1.0
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    height: int = 512
    width: int = 512
    batch_size: int = 4
    seed: int = 12345
    scheduler: str | None = "dpm++_2m"
    torch_dtype: str = "float16"
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT


def _prompt_for(job: GenerationJob, index: int) -> str:
    if job.prompts:
        return job.prompts[index % len(job.prompts)]
    return build_text_prompt(job.category, index=index)


def generate_synthetic_dataset(
    jobs: Sequence[GenerationJob],
    config: GenerationRunConfig,
) -> Path:
    """Produce the synthetic dataset described by ``jobs`` and return its root."""

    output_root = ensure_dir(config.output_dir)
    manifest_path = output_root / "manifest.csv"

    wrapper = StableDiffusionWrapper(
        DiffusionConfig(
            model_id=config.base_model_id,
            torch_dtype=config.torch_dtype,
            lora_weights=config.lora_weights,
            lora_scale=config.lora_scale,
            safety_checker=False,
            enable_attention_slicing=True,
            enable_vae_slicing=True,
        )
    )

    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "category", "prompt", "seed"])

        for job in jobs:
            cls_dir = ensure_dir(output_root / job.category)
            n_done = 0
            local_seed = config.seed
            logger.info("Generating %d images for category '%s'", job.num_images, job.category)
            while n_done < job.num_images:
                batch = min(config.batch_size, job.num_images - n_done)
                prompts = [_prompt_for(job, n_done + i) for i in range(batch)]
                params = GenerationParams(
                    prompt=prompts,
                    negative_prompt=[config.negative_prompt] * batch,
                    num_inference_steps=config.num_inference_steps,
                    guidance_scale=config.guidance_scale,
                    height=config.height,
                    width=config.width,
                    num_images_per_prompt=1,
                    seed=local_seed,
                    scheduler=config.scheduler,
                )
                images = wrapper.generate(params)
                for i, image in enumerate(images):
                    idx = n_done + i
                    filename = f"{job.category}_{idx:06d}.png"
                    image.save(cls_dir / filename)
                    writer.writerow([f"{job.category}/{filename}", job.category, prompts[i], local_seed])
                n_done += len(images)
                local_seed += 1

    logger.info("Synthetic dataset written to %s", output_root)
    return output_root


def expand_default_jobs(per_class: int) -> list[GenerationJob]:
    """Return one ``GenerationJob`` per supported disaster category."""

    return [
        GenerationJob(category=cat, num_images=per_class)
        for cat in DISASTER_PROMPT_TEMPLATES.keys()
    ]


__all__ = [
    "GenerationJob",
    "GenerationRunConfig",
    "expand_default_jobs",
    "generate_synthetic_dataset",
]
