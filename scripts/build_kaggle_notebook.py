"""Generate notebooks/kaggle_proxy_run.ipynb deterministically.

Run from the repo root: `python scripts/build_kaggle_notebook.py`.
We construct the notebook with `nbformat` instead of hand-writing JSON so cell
ordering, metadata and trust state stay consistent across regenerations.
"""

from __future__ import annotations

import nbformat as nbf
from pathlib import Path

REPO_URL = "https://github.com/Aristogiannis/uniwa-computer-vision.git"
OUTPUT = Path(__file__).resolve().parents[1] / "notebooks" / "kaggle_proxy_run.ipynb"

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
    "kaggle": {"accelerator": "gpu", "dataSources": [], "isInternetEnabled": True},
}


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text)


nb.cells = [
    md("""# Diffusion-based Satellite Augmentation — Kaggle proxy run

Full pipeline on a 3-class EuroSAT proxy (no xBD required):

1. Clone the repo + install
2. Download EuroSAT (HF parquet)
3. Extract to ImageFolder, upsample to 256 px
4. LoRA fine-tune **SD 1.5** on `forest` / `residential_buildings` / `river`
5. Generate synthetic samples per class
6. Three-arm downstream protocol: real-only / real+synthetic / synthetic-only
7. Emit metrics + qualitative grids into `/kaggle/working/outputs/`

**Hardware**: switch the kernel accelerator to **GPU P100** (Settings → Accelerator).
A run with `SMOKE = False` takes ~2 h on a P100; with `SMOKE = True` it finishes
in ~10 min and is meant to validate the wiring end-to-end before the long run."""),

    code("""# Toggle for a 10-min smoke validation vs the full ~2-h run.
SMOKE = True

# Proxy class set (must match prompt templates in cv_diffusion/preprocessing/prompts.py)
CLASSES = ["forest", "residential_buildings", "river"]

# Per-class training budget (downsampled from EuroSAT to keep LoRA fast & on-budget).
TRAIN_PER_CLASS = 800

# LoRA training resolution. SD 1.5 native is 512; 256 is the documented minimum.
LORA_RESOLUTION = 256

# Inference / generation resolution.
GEN_RESOLUTION = 256

# Synthetic images per class.
SYNTH_PER_CLASS = 200 if not SMOKE else 16

# LoRA step budget.
LORA_STEPS = 4000 if not SMOKE else 200

# Classifier arms epochs.
CLF_EPOCHS = 8 if not SMOKE else 1

print(f"SMOKE={SMOKE} | LORA_STEPS={LORA_STEPS} | SYNTH_PER_CLASS={SYNTH_PER_CLASS} | CLF_EPOCHS={CLF_EPOCHS}")"""),

    md("""## 1. Clone repo + install + GPU sanity check

**Set the kernel Accelerator to `GPU T4 x1`** (sm_75). P100 (sm_60) tripped
the LoRA step in earlier runs — the cause was the active PyTorch build not
shipping sm_60 kernels for some of the fp16 ops the diffusers trainer uses."""),
    code(f"""!git clone --depth 1 {REPO_URL} /kaggle/working/cv-diffusion
%cd /kaggle/working/cv-diffusion
!pip install -q --upgrade pip
# install the package itself but reuse Kaggle's preinstalled torch/numpy/pandas
!pip install -q -e . --no-deps
!pip install -q diffusers transformers accelerate peft safetensors \\
                huggingface-hub clean-fid timm pyarrow tifffile rasterio
import sys, subprocess

# Detect GPU via nvidia-smi BEFORE importing torch — so if it's a Pascal/P100
# (sm_60), we can swap to a sm_60-compatible torch wheel before any torch
# import binds the bad one into the kernel process.
smi = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True).strip()
print("nvidia-smi name:", smi)
print(subprocess.check_output(['nvidia-smi'], text=True))

PASCAL_MARKERS = ("P100", "P40", "P4", "Pascal")
if any(m in smi for m in PASCAL_MARKERS):
    print(f"\\n!!! {{smi}} detected (sm_60); Kaggle's torch 2.10+cu128 only ships sm_70+ kernels.")
    print("Installing torch 2.5.1+cu121 (last LTS with sm_60) ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "torch==2.5.1", "torchvision==0.20.1",
                           "--index-url", "https://download.pytorch.org/whl/cu121"])

import torch
cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None
print(f"\\ntorch: {{torch.__version__}}  cuda {{torch.version.cuda}}  device: {{smi}}  sm_{{cap[0]}}{{cap[1]}} " if cap else "no cuda")
if not torch.cuda.is_available():
    sys.exit("FATAL: no CUDA after torch import.")
# Sanity smoke: allocate a tiny tensor and run a matmul on GPU.
x = torch.randn(64, 64, device='cuda', dtype=torch.float16)
y = (x @ x).sum().item()
print(f"GPU smoke matmul OK: tr={{y:.4f}}")"""),

    md("""## 2. Download EuroSAT (parquet) and extract to ImageFolder

We pull the HF parquet shards once, then materialise PNGs into per-class
folders so the rest of the pipeline can use the standard ImageFolder API."""),

    code("""import io, json, re, os
from pathlib import Path
import pyarrow.parquet as pq
from PIL import Image
from huggingface_hub import snapshot_download

RAW_HF   = Path("/kaggle/working/eurosat_hf")
IMG_ROOT = Path("/kaggle/working/eurosat_images")  # 64x64 PNGs in class folders
IMG_ROOT.mkdir(parents=True, exist_ok=True)

snapshot_download(repo_id="blanchon/EuroSAT_RGB", repo_type="dataset",
                  local_dir=str(RAW_HF))

shards = ["train-00000-of-00001.parquet",
          "validation-00000-of-00001.parquet",
          "test-00000-of-00001.parquet"]

md = pq.read_metadata(RAW_HF / "data" / shards[0]).metadata
class_names = json.loads(md[b"huggingface"])["info"]["features"]["label"]["names"]
slug = {i: re.sub(r"[^a-z0-9]+", "_", n.lower()).strip("_") for i, n in enumerate(class_names)}
for s in slug.values():
    (IMG_ROOT / s).mkdir(exist_ok=True)

total = 0
for shard in shards:
    pf = pq.ParquetFile(RAW_HF / "data" / shard)
    for batch in pf.iter_batches(batch_size=1024, columns=["image", "label", "filename"]):
        imgs  = batch.column("image").to_pylist()
        labs  = batch.column("label").to_pylist()
        names = batch.column("filename").to_pylist()
        for img, lab, name in zip(imgs, labs, names):
            out = IMG_ROOT / slug[lab] / f"{Path(name).stem}.png"
            if not out.exists():
                Image.open(io.BytesIO(img["bytes"])).convert("RGB").save(out)
            total += 1
print(f"extracted {total} images into {IMG_ROOT}")
print({c: len(list((IMG_ROOT / c).iterdir())) for c in CLASSES})"""),

    md("""## 3. Build the proxy dataset

For each proxy class we:
* take a deterministic subset of `TRAIN_PER_CLASS` real images,
* upsample 64×64 → `LORA_RESOLUTION` for diffusion training,
* mirror the same images into a per-class folder consumed by the LoRA trainer."""),

    code("""import random
from PIL import Image

LORA_DATA = Path("/kaggle/working/proxy/lora_train")  # upsampled training images
for d in [LORA_DATA]:
    d.mkdir(parents=True, exist_ok=True)

rng = random.Random(42)
for cls in CLASSES:
    files = sorted((IMG_ROOT / cls).iterdir())
    rng.shuffle(files)
    files = files[:TRAIN_PER_CLASS]
    out_dir = LORA_DATA / cls
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        out = out_dir / f.name
        if out.exists():
            continue
        im = Image.open(f).convert("RGB").resize(
            (LORA_RESOLUTION, LORA_RESOLUTION), Image.BICUBIC)
        im.save(out)
    print(f"{cls}: {len(list(out_dir.iterdir()))} images at {LORA_RESOLUTION}px")"""),

    md("""## 4. LoRA fine-tune SD 1.5

Step count, resolution and seed are taken from the `SMOKE` toggle in cell 2.
The CLI streams its loss curve to `/kaggle/working/outputs/lora_train.log`
**and** we tee both stdout+stderr to `/kaggle/working/outputs/lora_traceback.log`
— so even if the kernel dies, the traceback survives on disk and the cell
output also prints the last 200 lines automatically on failure."""),

    code("""import subprocess
from pathlib import Path

Path("/kaggle/working/outputs").mkdir(parents=True, exist_ok=True)
TRACEBACK_LOG = "/kaggle/working/outputs/lora_traceback.log"

lora_cmd = (
    f"cv-train-lora "
    f"--train-data-dir /kaggle/working/proxy/lora_train "
    f"--output-dir /kaggle/working/outputs/lora/sd15_proxy "
    f"--pretrained-model-id stable-diffusion-v1-5/stable-diffusion-v1-5 "
    f"--resolution {LORA_RESOLUTION} "
    f"--max-train-steps {LORA_STEPS} "
    f"--train-batch-size 1 "
    f"--gradient-accumulation-steps 4 "
    f"--mixed-precision fp16 "
    f"--rank 8 --alpha 8 "
    f"--learning-rate 1e-4 "
    f"--seed 42 "
    f"--log-file /kaggle/working/outputs/lora_train.log"
)
print("Running:", lora_cmd)
print("Tracing  :", TRACEBACK_LOG)

# bash -c with pipefail so the pipe's exit code is meaningful.
full = f"set -o pipefail; {lora_cmd} 2>&1 | tee {TRACEBACK_LOG}"
result = subprocess.run(["bash", "-c", full])
rc = result.returncode
print(f"\\nLoRA exit code: {rc}")

if rc != 0:
    print(f"\\n=== last 200 lines of {TRACEBACK_LOG} ===")
    subprocess.run(["tail", "-n", "200", TRACEBACK_LOG])
    raise SystemExit(f"LoRA training failed (rc={rc}). Traceback log: {TRACEBACK_LOG}")"""),

    md("## 5. Generate synthetic images per class"),

    code("""import subprocess, shlex
SYNTH_ROOT = "/kaggle/working/proxy/synthetic"
cmd = (
    f"cv-generate "
    f"--output-dir {SYNTH_ROOT} "
    f"--lora-weights /kaggle/working/outputs/lora/sd15_proxy "
    f"--base-model-id stable-diffusion-v1-5/stable-diffusion-v1-5 "
    f"--per-class {SYNTH_PER_CLASS} "
    f"--steps 30 --guidance-scale 7.5 "
    f"--height {GEN_RESOLUTION} --width {GEN_RESOLUTION} "
    f"--batch-size 4 --seed 12345 "
    f"--torch-dtype float16 "
    + " ".join(f'--category {c}' for c in CLASSES)
)
print(cmd)
rc = subprocess.call(shlex.split(cmd))
print("generation exit code:", rc)
assert rc == 0
print({c: len(list((Path(SYNTH_ROOT)/c).iterdir())) for c in CLASSES})"""),

    md("## 6. Three-arm downstream protocol"),

    code("""# Build classifier splits from the FULL EuroSAT (only the 3 proxy classes)
import random, shutil
from pathlib import Path

CLF = Path("/kaggle/working/proxy/classifier")
if CLF.exists():
    shutil.rmtree(CLF)

rng2 = random.Random(42)
for cls in CLASSES:
    files = sorted((IMG_ROOT / cls).iterdir())
    rng2.shuffle(files)
    n = len(files)
    n_tr = int(round(n * 0.70))
    n_va = int(round(n * 0.15))
    splits = {"train": files[:n_tr],
              "val":   files[n_tr:n_tr + n_va],
              "test":  files[n_tr + n_va:]}
    for split, items in splits.items():
        d = CLF / split / cls
        d.mkdir(parents=True, exist_ok=True)
        for f in items:
            (d / f.name).symlink_to(f.resolve())
print({s: {c: len(list((CLF/s/c).iterdir())) for c in CLASSES}
       for s in ['train','val','test']})"""),

    code("""# Build the combined (real + synthetic) train root for arm 2.
COMBINED = Path("/kaggle/working/proxy/combined_train")
if COMBINED.exists():
    shutil.rmtree(COMBINED)
for cls in CLASSES:
    d = COMBINED / cls
    d.mkdir(parents=True, exist_ok=True)
    # symlink real train images
    for f in (CLF / "train" / cls).iterdir():
        (d / f.name).symlink_to(f.resolve())
    # copy synthetic images (resized to 64 px so they match the real data scale)
    synth_dir = Path(SYNTH_ROOT) / cls
    for s in synth_dir.iterdir():
        im = Image.open(s).convert("RGB").resize((64, 64), Image.BICUBIC)
        im.save(d / f"synth_{s.name}")
print({c: len(list((COMBINED/c).iterdir())) for c in CLASSES})"""),

    code("""# Resize synthetic-only into a parallel folder for arm 3.
SYNTH_ONLY = Path("/kaggle/working/proxy/synth_only_train")
if SYNTH_ONLY.exists():
    shutil.rmtree(SYNTH_ONLY)
for cls in CLASSES:
    d = SYNTH_ONLY / cls
    d.mkdir(parents=True, exist_ok=True)
    for s in (Path(SYNTH_ROOT) / cls).iterdir():
        im = Image.open(s).convert("RGB").resize((64, 64), Image.BICUBIC)
        im.save(d / s.name)
print({c: len(list((SYNTH_ONLY/c).iterdir())) for c in CLASSES})"""),

    code("""import subprocess, shlex
def run_arm(name, train_root):
    out_dir = f"/kaggle/working/outputs/arms/{name}"
    cmd = (f"cv-train-classifier "
           f"--train-root {train_root} "
           f"--val-root /kaggle/working/proxy/classifier/val "
           f"--test-root /kaggle/working/proxy/classifier/test "
           f"--output-dir {out_dir} "
           f"--backbone resnet18 "
           f"--epochs {CLF_EPOCHS} "
           f"--batch-size 64 "
           f"--image-size 64 "
           f"--log-file {out_dir}.log")
    print(cmd)
    rc = subprocess.call(shlex.split(cmd))
    assert rc == 0, f"arm {name} failed"
    return out_dir

real_only_dir   = run_arm("real_only",        "/kaggle/working/proxy/classifier/train")
real_synth_dir  = run_arm("real_plus_synth",  str(COMBINED))
synth_only_dir  = run_arm("synth_only",       str(SYNTH_ONLY))"""),

    md("## 7. Final results table"),

    code("""import json
def load(dir_): return json.load(open(Path(dir_)/"test_report.json"))
arms = {
    "1. real-only":         load(real_only_dir),
    "2. real + synthetic":  load(real_synth_dir),
    "3. synthetic-only":    load(synth_only_dir),
}
print(f"{'arm':24s} {'acc':>6s} {'mF1':>6s} {'wF1':>6s}")
for name, r in arms.items():
    print(f"{name:24s} {r['accuracy']:>6.3f} {r['macro_f1']:>6.3f} {r['weighted_f1']:>6.3f}")

d_acc = arms["2. real + synthetic"]["accuracy"] - arms["1. real-only"]["accuracy"]
d_f1  = arms["2. real + synthetic"]["macro_f1"] - arms["1. real-only"]["macro_f1"]
print()
print(f"Δ accuracy (arm2 − arm1) : {d_acc:+.4f} pp")
print(f"Δ macro-F1 (arm2 − arm1) : {d_f1:+.4f}")

# Persist summary for download
import json
summary = {name: {k: r[k] for k in ['accuracy','macro_f1','weighted_f1']}
           for name, r in arms.items()}
summary["delta_acc"] = d_acc
summary["delta_f1"]  = d_f1
out = Path("/kaggle/working/outputs/three_arm_summary.json")
out.write_text(json.dumps(summary, indent=2))
print(f"\\nwrote {out}")"""),

    md("## 8. Qualitative samples"),

    code("""import matplotlib.pyplot as plt
from PIL import Image as PILImage

fig, axes = plt.subplots(len(CLASSES), 6, figsize=(12, 2.2*len(CLASSES)))
for r, cls in enumerate(CLASSES):
    real_files  = sorted((CLF/'train'/cls).iterdir())[:3]
    synth_files = sorted((Path(SYNTH_ROOT)/cls).iterdir())[:3]
    for c, f in enumerate(real_files):
        axes[r, c].imshow(PILImage.open(f)); axes[r, c].set_axis_off()
    for c, f in enumerate(synth_files):
        axes[r, 3+c].imshow(PILImage.open(f)); axes[r, 3+c].set_axis_off()
    axes[r, 0].set_title(f"{cls} — real" if r==0 else "real", loc='left', fontsize=9)
    axes[r, 3].set_title("synthetic", loc='left', fontsize=9)
    axes[r, 0].text(-8, 32, cls.replace('_','\\n'), fontsize=9, ha='right', va='center')
fig.suptitle("Real (left 3) vs LoRA-generated synthetic (right 3)", y=0.995)
fig.tight_layout()
fig.savefig("/kaggle/working/outputs/qualitative.png", dpi=140, bbox_inches='tight')
plt.show()"""),

    md("""## 9. Download artifacts

After the run finishes, download `/kaggle/working/outputs/` from the Kaggle
file browser — it contains:

* `outputs/lora/sd15_proxy/` — LoRA adapter weights (`pytorch_lora_weights.safetensors`)
* `outputs/arms/*/test_report.json` — per-arm metrics
* `outputs/three_arm_summary.json` — headline table for the report
* `outputs/qualitative.png` — real-vs-synthetic grid
* `outputs/lora_train.log` — loss curve"""),
]

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.validate(nb)
nbf.write(nb, OUTPUT)
print(f"wrote {OUTPUT.relative_to(OUTPUT.parents[1])} with {len(nb.cells)} cells")
