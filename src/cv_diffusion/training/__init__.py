"""Training pipelines: LoRA fine-tuning + downstream classifier.

Lazy imports so importing the package never eagerly pulls in torch.
"""

_LAZY_NAMES = {
    "TrainLoRAConfig": "cv_diffusion.training.train_lora",
    "train_lora": "cv_diffusion.training.train_lora",
    "TrainClassifierConfig": "cv_diffusion.training.train_classifier",
    "train_classifier": "cv_diffusion.training.train_classifier",
}


def __getattr__(name: str):  # noqa: D401
    target = _LAZY_NAMES.get(name)
    if target is None:
        raise AttributeError(f"module 'cv_diffusion.training' has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(target), name)


__all__ = list(_LAZY_NAMES.keys())
