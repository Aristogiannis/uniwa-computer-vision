"""Evaluation utilities — image quality (FID/SSIM) and downstream classifier.

Lazy imports keep the package import cheap and dependency-free.
"""

_LAZY_NAMES = {
    "DownstreamProtocolResult": "cv_diffusion.evaluation.downstream",
    "run_downstream_protocol": "cv_diffusion.evaluation.downstream",
    "compute_fid": "cv_diffusion.evaluation.fid",
    "compute_pairwise_ssim": "cv_diffusion.evaluation.ssim",
    "compute_set_ssim": "cv_diffusion.evaluation.ssim",
}


def __getattr__(name: str):  # noqa: D401
    target = _LAZY_NAMES.get(name)
    if target is None:
        raise AttributeError(f"module 'cv_diffusion.evaluation' has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(target), name)


__all__ = list(_LAZY_NAMES.keys())
