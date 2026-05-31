# Methodology

This document describes, in publication-ready detail, the pipeline that the
code in `src/cv_diffusion/` implements. References point to
[`references.bib`](references.bib).

## 1. Research question

> *To what extent can diffusion models generate realistic synthetic
> satellite images for data augmentation in natural-disaster detection
> scenarios (floods, wildfires), addressing the shortage of labelled
> training data?*

The hypothesis is that fine-tuning a large pretrained latent-diffusion
model (Rombach et al., 2022) on a small disaster-specific corpus is
sufficient to produce synthetic tiles that meaningfully improve a
downstream disaster classifier, *without* requiring the full from-scratch
remote-sensing pretraining of DiffusionSat (Khanna et al., 2023).

## 2. Data

We use three publicly available satellite datasets, each playing a
different role.

| Dataset   | Role                                       | Size                       | License        |
|-----------|--------------------------------------------|----------------------------|----------------|
| EuroSAT   | Land-cover sanity check for the classifier | 27 000 images, 64 px       | MIT            |
| SEN12MS   | "Undamaged scene" prior (optional)         | 180 662 patches, 256 px    | CC BY 4.0      |
| xBD       | Primary disaster dataset, pre/post pairs   | ~22 000 pairs, 1024 px     | CC BY-NC-SA 4.0|

xBD is the only one of the three with explicit **disaster labels** (flood,
fire, hurricane, etc.) and **pre/post-event pairs**, so it carries the
main weight of the experiments. EuroSAT is used for a small auxiliary
classifier-only experiment in `notebooks/01_data_exploration.ipynb`.

### 2.1 Preprocessing

The pipeline (`cv_diffusion.preprocessing`) performs three steps:

1. **Percentile normalisation** (`normalize.percentile_normalize`).
   Per-channel clipping at the 2nd and 98th percentiles followed by
   min–max scaling to `[0, 1]`. This matches the recipe used by ESA SNAP
   and standard Sentinel viewers; it is robust to saturated outliers
   that would otherwise destroy a naïve min/max scaler.
2. **Tiling** (`tile.iter_tiles`). xBD scenes are 1024×1024 px and SD 1.5
   trains at 512×512 px. We use a sliding window of `size=stride=512`
   with `drop_partial=True` so every emitted tile is exactly square and
   the model never sees reflection-padded artefacts.
3. **Spectral alignment** (`spectral.rgb_from_sentinel2`). For SEN12MS the
   helper picks bands B04/B03/B02 (natural-colour) for the SD VAE; we
   also expose SWIR (B12/B11/B04) and false-colour NRG (B08/B04/B03)
   compositions because burnt areas are far more visible in SWIR. xBD is
   already RGB so spectral alignment is a no-op there.

## 3. Model

We build on **Stable Diffusion v1.5** (Rombach et al., 2022), keeping the
VAE and CLIP text encoder frozen. Only the UNet's cross-attention
projection matrices (`to_q`, `to_k`, `to_v`, `to_out.0`) receive **LoRA
adapters** (Hu et al., 2021) of rank 8 / alpha 8, dropout 0.

Rationale (see also the comments in `models/lora.py`):

* **LoRA, not full fine-tuning**: ~1.6 M trainable parameters vs ~860 M
  for the SD 1.5 UNet — fits a single 12-16 GB consumer GPU and yields
  ~25 MB adapter files.
* **UNet only, not text encoder**: the disaster prompts re-use everyday
  English ("flood", "wildfire", "satellite image"), so the CLIP text
  encoder already places them in sensible locations of latent space; a
  text-encoder LoRA marginally improves results in our pilot runs but
  doubles VRAM cost.
* **Why not DiffusionSat outright?** DiffusionSat (Khanna et al., 2023)
  is the strongest baseline for satellite generation, but reproducing it
  requires (a) 8 × A100 weeks of compute and (b) access to the fMoW /
  Satlas pretraining corpora. We instead borrow only its
  *metadata-embedding* idea conceptually: our caption template
  (`preprocessing/dataset.py::build_text_prompt`) encodes disaster
  category as a structured phrase rather than relying on free-form text.

### 3.1 Training recipe

We follow the official diffusers
[`train_text_to_image_lora.py`](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py)
recipe, simplified for readability:

* AdamW(β₁ = 0.9, β₂ = 0.999, weight-decay = 1e-2), lr = 1e-4 with cosine
  schedule, no warm-up.
* Loss: standard ε-prediction MSE between the predicted noise and the
  sampled `noise = N(0, I)`, with `DDPMScheduler` from the base model.
* Mixed precision: fp16 on CUDA, fp32 fallback on CPU/MPS.
* Gradient checkpointing on, gradient accumulation = 4, effective batch
  size 4.
* 4 000 update steps (≈ 100 epochs on ~1 000 disaster tiles).

Validation prompts are sampled every `--checkpointing-steps` and the loss
curve is dumped to `training_log.jsonl` for plotting.

## 4. Synthetic generation

`cv_diffusion.generation.generate_synthetic_dataset` loads the LoRA-fused
pipeline (`load_lora_weights` → `fuse_lora`) and produces N images per
category with the inference settings shown in `configs/generation.yaml`:

* DPM++ 2M Multistep scheduler, 30 steps, CFG 7.5.
* Negative prompt aimed at the typical SD failure modes on satellite
  imagery: cartoons, watermarks, faces, dramatic lighting.
* Deterministic per-image seed (`seed = base_seed + index`).
* Output mirrors the `SatelliteFolderDataset` layout so the classifier
  can ingest it directly.

## 5. Evaluation

### 5.1 Image quality

**FID** is computed with [`clean-fid`](https://github.com/GaParmar/clean-fid)
in `mode="clean"` (Parmar et al., 2022). We also expose `legacy_pytorch`
for comparison with older satellite-diffusion baselines that report
`pytorch-fid` numbers.

**SSIM** is computed with torchmetrics. Because we do not have a 1:1
correspondence between real and synthetic images, we report:

* `mean_ssim_all_pairs` — mean over the full real × fake Cartesian
  product (overall distributional proximity),
* `mean_nearest_ssim` — for each fake image, the max SSIM against any
  real image (memorisation / mode-collapse sanity check).

### 5.2 Downstream protocol (Frid-Adar et al., 2018)

The protocol implemented in `evaluation/downstream.py` is the canonical
three-arm comparison used throughout the synthetic-augmentation
literature:

1. **Real-only**: classifier trained on the real disaster tiles plus the
   classical augmentations (flips, crops, jitter) defined in
   `models/classifier.py::default_transforms`.
2. **Real + synthetic**: identical setup but the training set is the
   union of real and synthetic tiles (linked via symlinks under
   `outputs/downstream/combined_train/<class>`).
3. **Synthetic-only**: sanity-check arm; the classifier sees only
   synthetic data during training.

All three arms are evaluated on the **same real-only test set** with the
same seed. We report top-1 accuracy, macro-F1, weighted-F1 and per-class
recall, plus the deltas `acc(2) − acc(1)` and `f1(2) − f1(1)` — these
deltas are the headline numbers for the research question.

### 5.3 Statistical reporting

For the final report we re-run the protocol three times with seeds
`{42, 1337, 2024}` and report mean ± std on every metric. The CLI exposes
`--seed`; a small Bash for-loop is sufficient.

## 6. Compute budget

Reference timings on an NVIDIA RTX 4090 (24 GB) and a Kaggle P100 (16 GB):

| Stage           | RTX 4090 | Kaggle P100 |
|-----------------|----------|-------------|
| Preprocess (1 k scenes) | ~3 min | ~5 min |
| LoRA fine-tune (4 k steps) | ~25 min | ~95 min |
| Generation (600 images) | ~4 min | ~10 min |
| Classifier arm (20 epochs) | ~3 min | ~6 min |
| Full protocol (3 arms) | ~10 min | ~20 min |

Total wall-time of a full re-run on a free Kaggle P100 session is well
under the 12-hour notebook limit.

## 7. Threats to validity

* **Tiny domain corpus** — even with rank-8 LoRA, ~1 000 images is small
  by diffusion standards. FID values in the 30–80 range are typical
  for small-domain fine-tunes and are *not* directly comparable to
  ImageNet-scale FID. We rely on the *relative* trend
  (with-LoRA vs. base-SD) rather than absolute values.
* **Mode coverage** — diffusion fine-tunes can collapse onto whichever
  visual mode dominates the training set. The `mean_nearest_ssim`
  metric in §5.1 is our cheap detector for this; we also visualise the
  category-conditional samples in `notebooks/03_diffusion_generation.ipynb`.
* **Label leakage** — the synthetic dataset is generated from a model
  trained on the real labels. The downstream protocol therefore measures
  *augmentation value*, not generalisation; this is the same limitation
  faced by Frid-Adar et al. (2018) and is discussed in their §5.
