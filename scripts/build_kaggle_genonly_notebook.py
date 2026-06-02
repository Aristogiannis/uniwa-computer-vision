"""Generate notebooks/kaggle_genonly_run.ipynb deterministically.

This is a slim cousin of kaggle_proxy_run.ipynb that **skips the LoRA
fine-tune** and reuses the adapter at notebooks/lora_v7_4000steps/.
We only need ~2-3 hours of generation + classifier-arms — well inside
Kaggle's 9 h GPU session ceiling.

Tunes vs v7 that previously stalled in cv-generate on Pascal/P100:
- ``--torch-dtype float32``   (skip fp16; safer attention on P100)
- ``--batch-size 1``          (minimise VRAM pressure)
- ``--steps 20``              (33% fewer denoising steps)
- subprocess ``timeout`` wrapper so any future hang is bounded

Run from repo root:  python scripts/build_kaggle_genonly_notebook.py
"""

from __future__ import annotations

import nbformat as nbf
from pathlib import Path

REPO_URL = "https://github.com/Aristogiannis/uniwa-computer-vision.git"
OUTPUT = Path(__file__).resolve().parents[1] / "notebooks" / "genonly" / "kaggle_genonly_run.ipynb"

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
    md("""# Diffusion-based Satellite Augmentation — generate + 3-arm (v8)

Reuse the **4000-step LoRA adapter** from v7 and finish the pipeline:

1. Clone repo + install (with auto torch swap for Pascal sm_60)
2. Download EuroSAT (HF parquet) — same as before
3. Build the proxy train/val/test splits
4. **Skip LoRA fine-tune** — load `notebooks/lora_v7_4000steps/`
5. Generate synthetic images (float32, batch 1, 20 steps — bounded)
6. Build combined + synth-only train sets
7. Three-arm downstream protocol
8. Tar-gz `results.tar.gz` for one-shot local download

ETA on Kaggle P100: ~2-3 h."""),

    code("""# Per-class synthetic budget; classifier epochs per arm.
# v9 reduces scope after v7 (200 imgs/class, stalled silently) and v8
# (200 imgs/class @ float32, also stalled 9 h+ without surfaced log).
CLASSES = ["forest", "residential_buildings", "river"]
SYNTH_PER_CLASS = 50            # was 200 in v8 — quarter the work
GEN_RESOLUTION  = 256
GEN_STEPS       = 20            # was 30 in v7; 20 is a safe budget
GEN_BATCH       = 1             # was 4 in v7; 1 dodges VRAM edges on P100
GEN_DTYPE       = "float32"     # was float16 in v7; safer on Pascal sm_60
GEN_TIMEOUT_S   = 45 * 60       # was 90 m — tighter so a hang surfaces sooner
CLF_EPOCHS      = 8
print(f"CLASSES={CLASSES} | SYNTH_PER_CLASS={SYNTH_PER_CLASS} | "
      f"GEN_STEPS={GEN_STEPS} | GEN_BATCH={GEN_BATCH} | GEN_DTYPE={GEN_DTYPE} | "
      f"GEN_TIMEOUT_S={GEN_TIMEOUT_S} | CLF_EPOCHS={CLF_EPOCHS}")"""),

    md("""## 1. Clone repo + install + GPU sanity (with torch swap)"""),

    code(f"""!git clone --depth 1 {REPO_URL} /kaggle/working/cv-diffusion
%cd /kaggle/working/cv-diffusion
!pip install -q --upgrade pip
!pip install -q -e . --no-deps
!pip install -q diffusers transformers accelerate peft safetensors \\
                huggingface-hub clean-fid timm pyarrow tifffile rasterio

import sys, subprocess
# nvidia-smi probe BEFORE importing torch so we can swap to a sm_60-compatible
# wheel before any binding happens.
smi = subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'], text=True).strip()
print("device:", smi)
print(subprocess.check_output(['nvidia-smi'], text=True))

PASCAL_MARKERS = ("P100", "P40", "P4", "Pascal")
if any(m in smi for m in PASCAL_MARKERS):
    print(f"!!! {{smi}} (sm_60); installing torch 2.5.1+cu121 (last LTS with sm_60) ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "torch==2.5.1", "torchvision==0.20.1",
                           "--index-url", "https://download.pytorch.org/whl/cu121"])

import torch
cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None
print(f"torch: {{torch.__version__}}  cuda {{torch.version.cuda}}  sm_{{cap[0]}}{{cap[1]}}")
if not torch.cuda.is_available():
    sys.exit("FATAL: no CUDA")
x = torch.randn(64, 64, device='cuda', dtype=torch.float16)
print(f"GPU fp16 smoke: tr={{float((x @ x).sum()):.3f}}")"""),

    md("""## 2. Download + extract EuroSAT (same as v7)"""),

    code("""import io, json, re
from pathlib import Path
import pyarrow.parquet as pq
from PIL import Image
from huggingface_hub import snapshot_download

RAW_HF   = Path("/kaggle/working/eurosat_hf")
IMG_ROOT = Path("/kaggle/working/eurosat_images")
IMG_ROOT.mkdir(parents=True, exist_ok=True)
snapshot_download(repo_id="blanchon/EuroSAT_RGB", repo_type="dataset", local_dir=str(RAW_HF))

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
    for batch in pf.iter_batches(batch_size=1024, columns=["image","label","filename"]):
        imgs  = batch.column("image").to_pylist()
        labs  = batch.column("label").to_pylist()
        names = batch.column("filename").to_pylist()
        for img, lab, name in zip(imgs, labs, names):
            out = IMG_ROOT / slug[lab] / f"{Path(name).stem}.png"
            if not out.exists():
                Image.open(io.BytesIO(img["bytes"])).convert("RGB").save(out)
            total += 1
print(f"extracted {total} images")
print({c: len(list((IMG_ROOT / c).iterdir())) for c in CLASSES})"""),

    md("""## 3. Verify the bundled LoRA adapter is in place"""),

    code("""ADAPTER_DIR = Path("/kaggle/working/cv-diffusion/notebooks/lora_v7_4000steps")
print(f"adapter dir: {ADAPTER_DIR}")
for p in sorted(ADAPTER_DIR.iterdir()):
    print(f"  {p.name}  ({p.stat().st_size} bytes)")
assert (ADAPTER_DIR / "pytorch_lora_weights.safetensors").is_file(), \\
    "LoRA weights missing from cloned repo — was lora_v7_4000steps committed?" """),

    md("""## 4. Generate synthetic images per class

Wrapped in a `timeout` so any future hang is visible immediately."""),

    code("""import subprocess, shlex, time
from pathlib import Path

SYNTH_ROOT = "/kaggle/working/proxy/synthetic"
Path("/kaggle/working/outputs").mkdir(parents=True, exist_ok=True)
GEN_LOG = "/kaggle/working/outputs/gen.log"

cmd = (
    f"cv-generate "
    f"--output-dir {SYNTH_ROOT} "
    f"--lora-weights {ADAPTER_DIR} "
    f"--base-model-id stable-diffusion-v1-5/stable-diffusion-v1-5 "
    f"--per-class {SYNTH_PER_CLASS} "
    f"--steps {GEN_STEPS} --guidance-scale 7.5 "
    f"--height {GEN_RESOLUTION} --width {GEN_RESOLUTION} "
    f"--batch-size {GEN_BATCH} --seed 12345 "
    f"--torch-dtype {GEN_DTYPE} "
    + " ".join(f"--category {c}" for c in CLASSES)
)
print("Running:", cmd)
print("Logging :", GEN_LOG)

# Bounded subprocess: 90 min ceiling per the GEN_TIMEOUT_S knob above.
full = f"set -o pipefail; timeout {GEN_TIMEOUT_S}s {cmd} 2>&1 | tee {GEN_LOG}"
t0 = time.time()
result = subprocess.run(["bash", "-c", full])
elapsed = time.time() - t0
print(f"\\ngeneration exit code: {result.returncode}  wall time: {elapsed/60:.1f} min")

if result.returncode == 124:
    raise SystemExit(f"cv-generate hit {GEN_TIMEOUT_S}s timeout — log: {GEN_LOG}")
if result.returncode != 0:
    print(f"--- tail of {GEN_LOG} ---")
    subprocess.run(["tail", "-n", "120", GEN_LOG])
    raise SystemExit("cv-generate failed; see log tail above")

counts = {c: len(list((Path(SYNTH_ROOT)/c).iterdir())) for c in CLASSES}
print("synthetic counts:", counts)
assert all(v >= SYNTH_PER_CLASS for v in counts.values()), \\
    "generation produced fewer images than requested" """),

    md("""## 5. Build classifier splits"""),

    code("""import random, shutil

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
print({s: {c: len(list((CLF/s/c).iterdir())) for c in CLASSES} for s in ['train','val','test']})"""),

    code("""# Combined (real + synthetic) train root for arm 2
COMBINED = Path("/kaggle/working/proxy/combined_train")
if COMBINED.exists():
    shutil.rmtree(COMBINED)
for cls in CLASSES:
    d = COMBINED / cls
    d.mkdir(parents=True, exist_ok=True)
    for f in (CLF / "train" / cls).iterdir():
        (d / f.name).symlink_to(f.resolve())
    for s in (Path(SYNTH_ROOT) / cls).iterdir():
        im = Image.open(s).convert("RGB").resize((64, 64), Image.BICUBIC)
        im.save(d / f"synth_{s.name}")
print({c: len(list((COMBINED/c).iterdir())) for c in CLASSES})"""),

    code("""# Synthetic-only train root for arm 3 (downsampled to 64 px for fair comparison)
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

    md("""## 6. Three-arm downstream protocol"""),

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

real_only_dir   = run_arm("real_only",       "/kaggle/working/proxy/classifier/train")
real_synth_dir  = run_arm("real_plus_synth", str(COMBINED))
synth_only_dir  = run_arm("synth_only",      str(SYNTH_ONLY))"""),

    md("""## 7. Final results table"""),

    code("""import json
def load(d): return json.load(open(Path(d)/"test_report.json"))
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

summary = {name: {k: r[k] for k in ['accuracy','macro_f1','weighted_f1']}
           for name, r in arms.items()}
summary["delta_acc"] = d_acc
summary["delta_f1"]  = d_f1
out = Path("/kaggle/working/outputs/three_arm_summary.json")
out.write_text(json.dumps(summary, indent=2))
print(f"\\nwrote {out}")"""),

    md("""## 8. Qualitative samples"""),

    code("""import matplotlib.pyplot as plt
fig, axes = plt.subplots(len(CLASSES), 6, figsize=(12, 2.2*len(CLASSES)))
for r, cls in enumerate(CLASSES):
    real_files  = sorted((CLF/'train'/cls).iterdir())[:3]
    synth_files = sorted((Path(SYNTH_ROOT)/cls).iterdir())[:3]
    for c, f in enumerate(real_files):
        axes[r, c].imshow(Image.open(f));  axes[r, c].set_axis_off()
    for c, f in enumerate(synth_files):
        axes[r, 3+c].imshow(Image.open(f)); axes[r, 3+c].set_axis_off()
    if r == 0:
        axes[r, 0].set_title("forest — real" if False else "real", loc='left', fontsize=9)
        axes[r, 3].set_title("synthetic", loc='left', fontsize=9)
    axes[r, 0].text(-8, 32, cls.replace('_','\\n'), fontsize=9, ha='right', va='center')
fig.suptitle("Real (left 3) vs LoRA-generated synthetic (right 3) — v8, 4000-step LoRA", y=0.995)
fig.tight_layout()
fig.savefig("/kaggle/working/outputs/qualitative.png", dpi=140, bbox_inches='tight')
plt.show()"""),

    md("""## 9. Bundle results into one tar.gz"""),

    code("""import tarfile, os
ARCHIVE = "/kaggle/working/results.tar.gz"
PATHS = ["/kaggle/working/outputs", "/kaggle/working/proxy/synthetic"]
with tarfile.open(ARCHIVE, "w:gz") as tar:
    for p in PATHS:
        if os.path.exists(p):
            tar.add(p, arcname=os.path.basename(p))
print(f"wrote {ARCHIVE}  ({os.path.getsize(ARCHIVE) / 1e6:.1f} MB)")"""),
]

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.validate(nb)
nbf.write(nb, OUTPUT)
print(f"wrote {OUTPUT.relative_to(OUTPUT.parents[1])} with {len(nb.cells)} cells")
