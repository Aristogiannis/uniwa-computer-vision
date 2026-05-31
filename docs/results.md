# Results

This page is intentionally a **template**. Fill in the numbers once you
have run the full protocol on your hardware. The structure mirrors what
the final paper / poster should report.

## 1. Image-quality metrics

| Model variant                         | clean-FID ↓ | legacy-FID ↓ | Mean SSIM (set) ↑ | Mean nearest SSIM ↑ |
|---------------------------------------|-------------|--------------|-------------------|---------------------|
| SD 1.5 base (no LoRA)                 | _TBD_       | _TBD_        | _TBD_             | _TBD_               |
| SD 1.5 + LoRA (ours, rank 8)          | _TBD_       | _TBD_        | _TBD_             | _TBD_               |
| SDXL + LoRA (rank 16) — optional      | _TBD_       | _TBD_        | _TBD_             | _TBD_               |

Plotting helper: `notebooks/04_evaluation.ipynb` (cell "Figure 1").

## 2. Downstream protocol (xBD)

Backbone: ResNet-18 pretrained on ImageNet, 20 epochs, lr 3e-4, seed 42.
Synthetic images per class: 200.

| Arm                       | Top-1 Acc ↑ | Macro-F1 ↑ | Weighted-F1 ↑ |
|---------------------------|-------------|------------|----------------|
| 1. Real-only baseline     | _TBD_       | _TBD_      | _TBD_          |
| 2. Real + synthetic       | _TBD_       | _TBD_      | _TBD_          |
| 3. Synthetic-only         | _TBD_       | _TBD_      | _TBD_          |

Δ (arm 2 − arm 1): accuracy = _TBD_ pp, macro-F1 = _TBD_.

### Per-class recall

| Category       | Real-only | Real + synth | Δ      |
|----------------|-----------|--------------|--------|
| pre_disaster   | _TBD_     | _TBD_        | _TBD_  |
| flood          | _TBD_     | _TBD_        | _TBD_  |
| wildfire       | _TBD_     | _TBD_        | _TBD_  |

## 3. Ablations

Suggested ablations for the report (run them by overriding
`configs/lora_sd15.yaml`):

* **Rank sweep** — rank ∈ {4, 8, 16, 32} at fixed alpha = rank.
* **Step budget** — steps ∈ {1 k, 2 k, 4 k, 8 k}.
* **Synthetic budget** — synthetic-per-class ∈ {50, 100, 200, 400} for
  the downstream protocol.

## 4. Statistical significance

We re-run the full 3-arm protocol with seeds `{42, 1337, 2024}` and
report mean ± std. A simple paired t-test between real-only and real +
synthetic top-1 accuracy across the three seeds is the suggested
statistical test (see `scripts/run_pipeline.py` for the loop).

## 5. Qualitative samples

Place the figure used in the report at
`docs/figures/synthetic_grid.png`; the helper
`cv_diffusion.utils.io.save_image_grid` produces these grids directly
from a `data/synthetic/<category>/` folder.
