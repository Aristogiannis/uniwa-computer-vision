# Datasets

This project uses three publicly available satellite imagery datasets. All
three are research-only and free, but xBD requires registration before
download. The preprocessing pipeline expects the layout described below.

## EuroSAT — Land Use & Land Cover (Helber et al., 2019)

* **Source**: <https://github.com/phelber/EuroSAT>
* **HF dataset id**: `blanchon/EuroSAT_RGB` (or `blanchon/EuroSAT_MSI` for the 13-band MSI variant)
* **Content**: 27 000 Sentinel-2 patches at 64 × 64 px, 10 classes.
* **License**: MIT (code) + Copernicus open data (imagery).
* **Role in this project**: Sanity-check classifier dataset. Sentinel-2
  RGB statistics also serve as a "natural-scene" prior that helps the
  classifier generalise away from xBD's WorldView optics.
* **Caveat**: 64 px is far below SD 1.5's native resolution; do **not**
  use EuroSAT for LoRA fine-tuning without first upsampling to ≥ 256 px.

Layout expected by the preprocessor:

```
data/raw/eurosat/
├── AnnualCrop/
├── Forest/
├── …
└── SeaLake/
```

## SEN12MS — Multimodal Multispectral (Schmitt et al., 2019)

* **Source**: <https://mediatum.ub.tum.de/1474000>
* **arXiv**: [1906.07789](https://arxiv.org/abs/1906.07789)
* **Content**: 180 662 patch triplets of Sentinel-1 SAR (2 bands),
  Sentinel-2 multispectral (13 bands) and MODIS-derived land-cover, at
  256 × 256 px and 10 m GSD.
* **License**: CC BY 4.0.
* **Role in this project**: Optional source of "undamaged scene" tiles
  for the `pre_disaster` category — useful if you want to balance the
  pre/post counts in the xBD training set.
* **Caveat**: Full archive is ~430 GB. Download a single season ROI
  (~30 GB) and extract under `data/raw/sen12ms/` keeping the directory
  structure. The `cv_diffusion.preprocessing.spectral` module handles
  the band selection.

## xBD — Damage Assessment (Gupta et al., 2019)

* **Source**: <https://xview2.org/>
* **arXiv**: [1911.09296](https://arxiv.org/abs/1911.09296)
* **Content**: ~22 000 pre/post co-registered Maxar WorldView-2/3 image
  pairs at 1024 × 1024 px, ~0.5 m GSD, with GeoJSON building polygons
  and a 4-class damage scale (no-damage / minor / major / destroyed).
  Disaster events include flood, wildfire, earthquake, hurricane,
  tsunami, volcanic eruption.
* **License**: CC BY-NC-SA 4.0 (non-commercial).
* **Role in this project**: Primary dataset. Provides both the
  fine-tuning images and the downstream classifier labels.
* **Access**:

  1. Register at <https://xview2.org/>.
  2. Sign the data-use agreement.
  3. Download the Tier1, Tier3, Test and Holdout tar archives.
  4. Extract into `data/raw/xbd/` so the layout looks like:

     ```
     data/raw/xbd/
     ├── train/
     │   ├── images/
     │   │   ├── hurricane-florence_00000000_pre_disaster.png
     │   │   ├── hurricane-florence_00000000_post_disaster.png
     │   │   └── …
     │   └── labels/                # GeoJSON
     ├── tier3/
     ├── test/
     └── hold/
     ```

After extracting, run the preprocessor:

```bash
cv-preprocess --config configs/preprocess.yaml
```

This produces the disaster-category folder layout expected by the LoRA
trainer and the classifier (the category is parsed from the xBD filename
prefix, e.g. `wildfire_california_…_post_disaster.png`).

## Splits used by the downstream protocol

The downstream protocol expects three folders with the same set of class
subdirectories:

```
data/processed/xbd_classifier/
├── train/
│   ├── pre_disaster/
│   ├── flood/
│   └── wildfire/
├── val/
│   └── …
└── test/
    └── …
```

A small helper notebook (`notebooks/01_data_exploration.ipynb`) shows how
to derive these splits from the raw xBD archives using stratified
sampling. We default to a 70 / 15 / 15 split with a fixed seed so the
test set is identical across the three protocol arms.
