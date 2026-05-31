"""Synthetic image generation pipelines built on the fine-tuned LoRA."""

_LAZY_NAMES = {
    "GenerationJob": "cv_diffusion.generation.generate",
    "GenerationRunConfig": "cv_diffusion.generation.generate",
    "expand_default_jobs": "cv_diffusion.generation.generate",
    "generate_synthetic_dataset": "cv_diffusion.generation.generate",
}


def __getattr__(name: str):  # noqa: D401
    target = _LAZY_NAMES.get(name)
    if target is None:
        raise AttributeError(f"module 'cv_diffusion.generation' has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(target), name)


__all__ = list(_LAZY_NAMES.keys())
