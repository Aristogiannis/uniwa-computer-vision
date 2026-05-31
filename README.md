# Diffusion Models for Image Generation — Synthetic Satellite Augmentation for Disaster Detection

**University of West Attica · Computer Vision course project**

> *To what extent can diffusion models generate realistic synthetic satellite
> images for data augmentation in natural-disaster detection scenarios
> (floods, wildfires), addressing the shortage of labelled training data?*

This repository implements a complete pipeline that answers the question
above by (1) fine-tuning Stable Diffusion v1.5 on real satellite tiles from
xBD with **LoRA** adapters conditioned on disaster category, (2) generating
synthetic disaster imagery with the fine-tuned model, and (3) measuring
both the visual quality of the synthetic images (FID, SSIM) and their
**downstream utility** as augmentation data for a disaster classifier
(real-only vs. real + synthetic vs. synthetic-only).

---

## 1. Project at a glance

| Stage | What it does | Entry point |
|-------|--------------|-------------|
| 1. Preprocess | Normalises and tiles raw xBD / SEN12MS / EuroSAT scenes into 512×512 RGB tiles, organised into per-category folders. | `cv-preprocess` / `scripts/run_pipeline.py --preprocess` |
| 2. LoRA fine-tune | Trains LoRA adapters on top of frozen SD 1.5 with category-aware text prompts. | `cv-train-lora` |
| 3. Synthetic generation | Samples N synthetic tiles per category using the LoRA-fused pipeline, writing an ImageFolder + manifest. | `cv-generate` |
| 4. Evaluation | Computes clean-FID and SSIM, then runs the 3-arm classifier protocol. | `cv-evaluate fid` / `ssim` / `downstream` |

The code is intentionally small (~2 k LOC), pure Python, and reproduces the
methodology described in the assignment brief.

## 2. Repository layout

```
.
├── configs/                  # YAML configs for each stage
├── data/
│   ├── raw/                  # Where you place EuroSAT / SEN12MS / xBD
│   ├── processed/            # Tiled, normalised training images
│   └── synthetic/            # Output of cv-generate
├── docs/                     # Methodology, datasets, references
├── notebooks/                # Exploration / demo notebooks
├── scripts/                  # download_data.sh, run_pipeline.py
├── src/cv_diffusion/         # The Python package
│   ├── preprocessing/        # Normalization, tiling, spectral alignment, datasets
│   ├── models/               # SD wrapper, LoRA, classifier
│   ├── training/             # train_lora.py, train_classifier.py
│   ├── generation/           # Synthetic image generation
│   ├── evaluation/           # FID, SSIM, downstream 3-arm protocol
│   ├── cli/                  # Console entry points
│   └── utils/                # Logging, IO, seeding, config
└── tests/                    # pytest smoke tests
```

## 3. Quick start

### 3.1 Install

```bash
# Python 3.10+ recommended. A GPU is needed for training; inference works on CPU
# but is very slow.
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

### 3.2 Get the data

We use **xBD** as the primary disaster dataset (it's the only one of the
three with pre/post-event pairs and disaster labels), plus EuroSAT as a
sanity-check classifier dataset.

```bash
scripts/download_data.sh eurosat   # ~2 GB, fully automatic
scripts/download_data.sh xbd       # prints registration instructions
scripts/download_data.sh sen12ms   # prints download instructions
```

xBD requires registering at <https://xview2.org/> and signing the
non-commercial data-use agreement. See `docs/datasets.md` for the full
layout the preprocessing stage expects.

### 3.3 Preprocess

```bash
cv-preprocess --config configs/preprocess.yaml
```

This turns the raw xBD scenes into a folder of 512×512 tiles. By default
tiles are grouped into `pre_disaster/`, `flood/`, `wildfire/` etc. so they
can be consumed directly by both the LoRA trainer and the classifier.

### 3.4 Fine-tune the diffusion model

```bash
cv-train-lora --config configs/lora_sd15.yaml
```

Defaults to SD 1.5 at 512 px, rank 8, lr 1e-4, 4 000 update steps. Fits in
**12 GB** of VRAM with `--mixed-precision fp16` and grad checkpointing. An
optional `configs/lora_sdxl.yaml` is provided for users with ≥16 GB cards.

The trainer writes LoRA adapter weights to `outputs/lora/sd15_disaster/`
plus a `training_log.jsonl` curve and a TensorBoard log directory.

### 3.5 Generate synthetic disaster images

```bash
cv-generate --config configs/generation.yaml
```

Produces `data/synthetic/<category>/<category>_NNNNNN.png` and a
`manifest.csv` recording the prompt and seed for every image.

### 3.6 Evaluate

```bash
# Image-quality metrics
cv-evaluate fid  --real-dir data/processed/xbd_tiles/flood \
                 --fake-dir data/synthetic/flood --mode clean
cv-evaluate ssim --real-dir data/processed/xbd_tiles/flood \
                 --fake-dir data/synthetic/flood

# 3-arm downstream protocol
cv-evaluate downstream \
    --real-train-root data/processed/xbd_classifier/train \
    --real-val-root   data/processed/xbd_classifier/val \
    --real-test-root  data/processed/xbd_classifier/test \
    --synthetic-root  data/synthetic \
    --output-root     outputs/downstream
```

The downstream command trains three classifiers in sequence
(real-only, real + synthetic, synthetic-only) and writes a JSON summary
with per-arm accuracy / macro-F1 and the delta between arms 1 and 2.

### 3.7 One-shot smoke test

```bash
python scripts/run_pipeline.py --all \
    --real-train-root data/processed/xbd_classifier/train \
    --real-test-root  data/processed/xbd_classifier/test
```

## 4. Hyperparameters at a glance

| Setting | Value | Source |
|---------|-------|--------|
| Base model | `runwayml/stable-diffusion-v1-5` | Rombach et al. 2022 |
| LoRA rank / alpha | 8 / 8 | diffusers default, validated for small-domain LoRA |
| Learning rate | 1 e-4 (AdamW) | diffusers `train_text_to_image_lora.py` |
| Train batch size | 1 × grad-accum 4 = 4 effective | fits a 12 GB GPU at 512 px fp16 |
| Steps | 4 000 | rule of thumb for a few-thousand-image domain dataset |
| Scheduler (inference) | DPM++ 2M | fastest acceptable-quality sampler in diffusers |
| Guidance scale | 7.5 | SD 1.5 default |
| Inference steps | 30 | DPM++ 2M converges by ~25 steps |
| Classifier backbone | ResNet-18 from `timm` (ImageNet-pretrained) | fast on T4 / P100 |
| Classifier image size | 224 px | matches ImageNet stats |

Full discussion lives in [`docs/methodology.md`](docs/methodology.md).

## 5. Reproducibility

* Every stage writes a `config.json` and a structured log to its output
  directory.
* All RNGs are seeded via `cv_diffusion.utils.seed.seed_everything(...)`.
* Synthetic samples carry their generation seed in `manifest.csv` so any
  single image can be regenerated independently.

## 6. Tests

```bash
pytest -m "not slow"
```

The default suite is CPU-only and finishes in seconds; tests marked `slow`
or `gpu` are skipped unless explicitly selected.

## 7. References

See [`docs/references.bib`](docs/references.bib). The key citations are:

* Rombach et al. (2022). *High-Resolution Image Synthesis with Latent Diffusion Models.* CVPR.
* Khanna et al. (2023). *DiffusionSat: A Generative Foundation Model for Satellite Imagery.* arXiv:2312.03606.
* Gupta et al. (2019). *xBD: A Dataset for Assessing Building Damage from Satellite Imagery.* CVPR Workshops.
* Frid-Adar et al. (2018). *GAN-based Synthetic Medical Image Augmentation…* *Neurocomputing* 321.
* Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685.
* Parmar et al. (2022). *On Aliased Resizing and Surprising Subtleties in GAN Evaluation.* CVPR.

## 8. License

MIT (this code). Note that the **xBD dataset is CC BY-NC-SA 4.0**, i.e.
non-commercial, share-alike. Any models or synthetic data derived from it
inherit those restrictions.
