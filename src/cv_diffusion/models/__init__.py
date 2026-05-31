"""Model wrappers: Stable Diffusion + LoRA, plus a disaster classifier.

Imports are lazy so that the package can be used in environments where
PyTorch is not installed (e.g. the CI smoke tests for CLI ``--help``).
``LoRAConfig`` is a pure-Python dataclass and is therefore available
eagerly.
"""

from cv_diffusion.models.lora import LoRAConfig

_LAZY_NAMES = {
    "DisasterClassifier": "cv_diffusion.models.classifier",
    "build_classifier": "cv_diffusion.models.classifier",
    "StableDiffusionWrapper": "cv_diffusion.models.diffusion",
    "apply_lora_to_unet": "cv_diffusion.models.lora",
}


def __getattr__(name: str):  # noqa: D401
    target = _LAZY_NAMES.get(name)
    if target is None:
        raise AttributeError(f"module 'cv_diffusion.models' has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(target), name)


__all__ = [
    "DisasterClassifier",
    "LoRAConfig",
    "StableDiffusionWrapper",
    "apply_lora_to_unet",
    "build_classifier",
]
