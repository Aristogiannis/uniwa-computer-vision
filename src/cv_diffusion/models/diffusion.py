"""Thin wrappers around the ``diffusers`` Stable Diffusion pipeline.

We avoid re-implementing the well-tested ``StableDiffusionPipeline`` — the
wrapper exists only to:

* centralise the choice of base model id (so the rest of the codebase doesn't
  hard-code "stable-diffusion-v1-5/stable-diffusion-v1-5"),
* hide the awkward eager vs xformers attention switch behind a flag, and
* offer a friendly ``generate(...)`` method that returns a list of PIL images.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from cv_diffusion.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"


@dataclass
class DiffusionConfig:
    """Subset of pipeline settings we expose."""

    model_id: str = DEFAULT_MODEL_ID
    revision: str | None = None
    torch_dtype: str = "float16"  # or "float32"
    safety_checker: bool = False
    enable_xformers: bool = False
    enable_attention_slicing: bool = True
    enable_vae_slicing: bool = True
    enable_cpu_offload: bool = False
    lora_weights: str | None = None  # Path or HF repo id to a LoRA adapter
    lora_scale: float = 1.0


@dataclass
class GenerationParams:
    """User-facing inference parameters."""

    prompt: str | Sequence[str] = ""
    negative_prompt: str | Sequence[str] | None = None
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    height: int = 512
    width: int = 512
    num_images_per_prompt: int = 1
    seed: int | None = None
    scheduler: str | None = None  # e.g. "dpm++_2m", "euler_a"
    metadata: dict = field(default_factory=dict)


def _select_dtype(name: str):
    import torch

    return {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }[name.lower()]


def _build_scheduler(pipe, name: str):
    """Swap the default scheduler in-place. Best-effort — unknown names raise."""

    from diffusers import (
        DDIMScheduler,
        DPMSolverMultistepScheduler,
        EulerAncestralDiscreteScheduler,
        EulerDiscreteScheduler,
        PNDMScheduler,
    )

    schedulers = {
        "ddim": DDIMScheduler,
        "pndm": PNDMScheduler,
        "euler": EulerDiscreteScheduler,
        "euler_a": EulerAncestralDiscreteScheduler,
        "dpm++_2m": DPMSolverMultistepScheduler,
        "dpmpp_2m": DPMSolverMultistepScheduler,
    }
    key = name.lower()
    if key not in schedulers:
        raise ValueError(f"Unknown scheduler '{name}'. Options: {sorted(schedulers)}")
    pipe.scheduler = schedulers[key].from_config(pipe.scheduler.config)
    return pipe


class StableDiffusionWrapper:
    """Lazy-loading wrapper around ``StableDiffusionPipeline``.

    The underlying pipeline is loaded on first call to :meth:`pipeline` or
    :meth:`generate` so that instantiating the wrapper itself is cheap (useful
    for tests and CLI ``--help`` calls).
    """

    def __init__(self, config: DiffusionConfig | None = None) -> None:
        self.config = config or DiffusionConfig()
        self._pipeline = None
        self._device: str | None = None

    @property
    def device(self) -> str:
        if self._device is not None:
            return self._device
        import torch

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        return self._device

    @property
    def pipeline(self):
        if self._pipeline is None:
            self._pipeline = self._build_pipeline()
        return self._pipeline

    def _build_pipeline(self):
        from diffusers import StableDiffusionPipeline

        cfg = self.config
        logger.info("Loading Stable Diffusion: %s (dtype=%s)", cfg.model_id, cfg.torch_dtype)

        dtype = _select_dtype(cfg.torch_dtype) if self.device == "cuda" else _select_dtype("float32")
        kwargs = dict(torch_dtype=dtype, revision=cfg.revision)
        if not cfg.safety_checker:
            kwargs.update(safety_checker=None, requires_safety_checker=False)

        pipe = StableDiffusionPipeline.from_pretrained(cfg.model_id, **kwargs)
        pipe = pipe.to(self.device)

        if cfg.enable_xformers:
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception as exc:  # pragma: no cover - environment-dependent
                logger.warning("xformers unavailable, falling back: %s", exc)

        if cfg.enable_attention_slicing:
            pipe.enable_attention_slicing()
        if cfg.enable_vae_slicing:
            pipe.enable_vae_slicing()
        if cfg.enable_cpu_offload and self.device == "cuda":
            pipe.enable_model_cpu_offload()

        if cfg.lora_weights:
            logger.info("Loading LoRA weights from %s (scale=%s)", cfg.lora_weights, cfg.lora_scale)
            pipe.load_lora_weights(cfg.lora_weights)
            try:
                pipe.fuse_lora(lora_scale=cfg.lora_scale)
            except Exception:
                # Some diffusers versions auto-apply on load; ignore.
                pass

        return pipe

    def generate(self, params: GenerationParams):
        """Run inference with the given parameters and return PIL images."""

        import torch

        pipe = self.pipeline
        if params.scheduler:
            _build_scheduler(pipe, params.scheduler)

        generator = None
        if params.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(params.seed)

        output = pipe(
            prompt=list(params.prompt) if isinstance(params.prompt, Iterable) and not isinstance(params.prompt, str) else params.prompt,
            negative_prompt=list(params.negative_prompt) if isinstance(params.negative_prompt, Iterable) and not isinstance(params.negative_prompt, str) and params.negative_prompt is not None else params.negative_prompt,
            num_inference_steps=params.num_inference_steps,
            guidance_scale=params.guidance_scale,
            height=params.height,
            width=params.width,
            num_images_per_prompt=params.num_images_per_prompt,
            generator=generator,
        )
        return list(output.images)

    def save_pretrained(self, output_dir: str | Path) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.pipeline.save_pretrained(out)
