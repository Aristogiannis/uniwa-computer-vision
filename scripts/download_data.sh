#!/usr/bin/env bash
# Convenience helpers for downloading the three datasets used by the project.
# Each dataset has a different access policy — read the comments before running.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ROOT}/data/raw"
mkdir -p "${DATA_DIR}"

usage() {
    cat <<'USAGE'
Usage: download_data.sh <dataset>
  eurosat   Download EuroSAT RGB (MIT, ~2 GB).
  sen12ms   Print TUM mediaTUM download instructions (manual).
  xbd       Print xView2.org registration instructions (manual).
USAGE
}

case "${1:-}" in
    eurosat)
        cd "${DATA_DIR}"
        if [ -d eurosat ]; then
            echo "EuroSAT already present at ${DATA_DIR}/eurosat" >&2
        else
            python - <<'PY'
from pathlib import Path
from huggingface_hub import snapshot_download
out = Path("eurosat")
snapshot_download(
    repo_id="blanchon/EuroSAT_RGB",
    repo_type="dataset",
    local_dir=str(out),
    local_dir_use_symlinks=False,
)
print(f"Downloaded EuroSAT RGB to {out.resolve()}")
PY
        fi
        ;;
    sen12ms)
        cat <<'MSG'
SEN12MS is hosted on TUM mediaTUM and must be downloaded manually:
  https://mediatum.ub.tum.de/1474000
The full dataset is ~430 GB. For LoRA fine-tuning it is sufficient to
download a single season summer/spring/autumn ROI archive (~30 GB) and
extract it under data/raw/sen12ms/.
MSG
        ;;
    xbd)
        cat <<'MSG'
xBD is distributed by the xView2 challenge and requires registration:
  1. Create an account at https://xview2.org/
  2. Sign the data-use agreement (CC BY-NC-SA 4.0).
  3. Download the Tier1, Tier3, Test and Holdout tar archives.
  4. Extract them under data/raw/xbd/ keeping the directory layout
     (post/ pre/ images/labels).
MSG
        ;;
    *)
        usage
        exit 1
        ;;
esac
