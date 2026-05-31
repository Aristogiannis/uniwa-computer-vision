"""Diffusion-based satellite image generation for natural-disaster data augmentation.

Public API surface for the ``cv_diffusion`` package. Importing the top-level
package is intentionally cheap: heavy dependencies (PyTorch, diffusers, ...) are
imported lazily inside submodules.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("cv-diffusion")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = ["__version__"]
