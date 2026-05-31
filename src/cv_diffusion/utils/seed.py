"""Reproducibility helpers."""

from __future__ import annotations

import os
import random


def seed_everything(seed: int = 42, *, deterministic: bool = False) -> int:
    """Seed Python, NumPy, and PyTorch RNGs.

    Parameters
    ----------
    seed:
        Base seed. Also written into ``PYTHONHASHSEED``.
    deterministic:
        If ``True``, request CuDNN determinism. Slower; intended for the
        final reproducibility runs reported in the paper, not exploratory work.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # numpy is optional at import time
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    return seed
