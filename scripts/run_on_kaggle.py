"""Push and run a Kaggle notebook from this device using REST + Bearer auth.

The published kaggle CLI (v1.7.4.5) still expects the legacy kaggle.json
username/password Basic auth and cannot consume the new ``KGAT_*`` access
tokens. The REST API itself accepts ``Authorization: Bearer KGAT_*``, so
this script talks to ``/api/v1/kernels/{push,status,output}`` directly.

Usage::

    KAGGLE_API_TOKEN="$(cat ~/.kaggle/access_token)" \\
        python scripts/run_on_kaggle.py push-and-wait \\
            --folder notebooks \\
            --output-dir outputs/kaggle_run

Sub-commands:
  push                Push the notebook (one version) and exit
  status              Print the latest run status as JSON
  output              Download the latest run output to --output-dir
  push-and-wait       push -> poll until terminal status -> output
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests

API_ROOT = "https://www.kaggle.com/api/v1"
DEFAULT_POLL_INTERVAL = 30
DEFAULT_TIMEOUT = 60 * 60 * 4  # 4 h ceiling


def _token() -> str:
    tok = os.environ.get("KAGGLE_API_TOKEN")
    if tok:
        return tok.strip()
    p = Path.home() / ".kaggle" / "access_token"
    if p.is_file():
        return p.read_text().strip()
    raise SystemExit(
        "No Kaggle token. Set KAGGLE_API_TOKEN or write ~/.kaggle/access_token"
    )


def _session() -> requests.Session:
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {_token()}"
    s.headers["Accept"] = "application/json"
    s.headers["User-Agent"] = "cv-diffusion-kaggle-runner/0.1"
    return s


def _build_push_payload(folder: Path) -> tuple[dict[str, Any], str]:
    meta = json.loads((folder / "kernel-metadata.json").read_text())
    code_file = folder / meta["code_file"]
    nb = json.loads(code_file.read_text())
    # Strip stale outputs so the push diff is minimal and clean.
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        # Kaggle expects a single string source, not a list of lines.
        if isinstance(cell.get("source"), list):
            cell["source"] = "".join(cell["source"])

    slug = meta["id"]
    payload = {
        "slug": slug,
        "newTitle": meta.get("title"),
        "text": json.dumps(nb),
        "language": meta.get("language", "python"),
        "kernelType": meta.get("kernel_type", "notebook"),
        "isPrivate": str(meta.get("is_private", "true")).lower() == "true",
        "enableGpu": str(meta.get("enable_gpu", "true")).lower() == "true",
        "enableTpu": str(meta.get("enable_tpu", "false")).lower() == "true",
        "enableInternet": str(meta.get("enable_internet", "true")).lower() == "true",
        "datasetDataSources": meta.get("dataset_sources", []),
        "competitionDataSources": meta.get("competition_sources", []),
        "kernelDataSources": meta.get("kernel_sources", []),
        "modelDataSources": meta.get("model_sources", []),
        "categoryIds": meta.get("keywords", []),
    }
    return payload, slug


def push(folder: Path) -> dict[str, Any]:
    payload, slug = _build_push_payload(folder)
    sess = _session()
    r = sess.post(f"{API_ROOT}/kernels/push", json=payload, timeout=120)
    if r.status_code >= 400:
        raise SystemExit(f"push failed: HTTP {r.status_code}\n{r.text[:2000]}")
    body = r.json()
    # Trust Kaggle's `ref` ("/code/<owner>/<slug>") over the metadata id —
    # Kaggle slugifies the title and ignores the metadata id when they
    # disagree, then later API calls 403 against the metadata id.
    ref = body.get("ref") or body.get("Ref") or ""
    actual = ref.lstrip("/")
    if actual.startswith("code/"):
        actual = actual[len("code/"):]
    body["_slug"] = actual or slug
    return body


def status(slug: str) -> dict[str, Any]:
    user, k = slug.split("/", 1)
    # Retry transient connection failures (TLS drops, DNS hiccups, brief
    # offline periods) generously — the kernel is running on Kaggle's side
    # and there's no reason to abandon the poll because of local-side flakes.
    # Budget: 30 attempts capped at 5 min apart = ~tolerates ~2 h of network
    # weather. Enough for a multi-hour LoRA run.
    last_exc: Exception | None = None
    for i in range(30):
        try:
            r = _session().get(
                f"{API_ROOT}/kernels/status",
                params={"user_name": user, "kernel_slug": k},
                timeout=30,
            )
            if r.status_code >= 400:
                raise SystemExit(f"status failed: HTTP {r.status_code}\n{r.text[:2000]}")
            return r.json()
        except (requests.exceptions.SSLError,
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as e:
            last_exc = e
            wait = min(300, 2 ** (i + 2))
            print(f"  status transient {type(e).__name__}; retry in {wait}s", flush=True)
            time.sleep(wait)
    raise SystemExit(f"status kept failing after retries: {last_exc}")


def _get_with_retry(sess: requests.Session, url: str, params: dict, attempts: int = 6):
    last_text = ""
    last_exc: Exception | None = None
    for i in range(attempts):
        try:
            r = sess.get(url, params=params, timeout=60)
        except (requests.exceptions.SSLError,
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as e:
            last_exc = e
            wait = min(60, 2 ** (i + 2))
            print(f"  GET {url} -> {type(e).__name__}; retry in {wait}s")
            time.sleep(wait)
            continue
        if r.status_code < 400:
            return r
        last_text = r.text[:400]
        # Kaggle's edge rate-limits aggressively after a burst of probes and
        # falls through to a generic 404 HTML page once 429 quota is exhausted.
        # Either way the recovery is the same: wait long enough for the window
        # to reset (~60 s) and try once. Exponential backoff capped at 120 s.
        wait = 60 if r.status_code == 429 else min(120, 2 ** (i + 3))
        print(f"  GET {url} -> HTTP {r.status_code}; retry in {wait}s")
        time.sleep(wait)
    raise SystemExit(f"GET {url} kept failing: {last_text or last_exc}")


def download_output(slug: str, out_dir: Path) -> None:
    user, k = slug.split("/", 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    sess = _session()
    page_token = ""
    files: list[dict] = []
    log_text: str | None = None
    while True:
        params = {"user_name": user, "kernel_slug": k, "page_size": 200}
        if page_token:
            params["page_token"] = page_token
        r = _get_with_retry(sess, f"{API_ROOT}/kernels/output", params)
        body = r.json()
        batch = body.get("files") if isinstance(body, dict) else body
        if batch:
            files.extend(batch)
        if log_text is None and isinstance(body, dict):
            log_text = (
                body.get("log")
                or body.get("logNullable")
                or body.get("Log")
            )
        page_token = body.get("nextPageToken") if isinstance(body, dict) else ""
        if not page_token:
            break
    print(f"  {len(files)} output files")
    for f in files:
        url = f.get("url") or f.get("downloadUrl")
        name = f.get("fileName") or f.get("name") or f.get("filename")
        if not url or not name:
            continue
        dest = out_dir / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        with sess.get(url, stream=True, timeout=300) as rr:
            rr.raise_for_status()
            with dest.open("wb") as fh:
                for chunk in rr.iter_content(chunk_size=64 * 1024):
                    fh.write(chunk)
        print(f"  - {name} ({dest.stat().st_size} bytes)")
    if log_text:
        # Kaggle returns the log as a JSON-encoded list of stream entries.
        # Try to decode -> plain text; if that fails, save the raw envelope.
        plain: str
        try:
            entries = json.loads(log_text)
            plain = "".join(e.get("data", "") for e in entries)
        except Exception:
            plain = log_text
        log_path = out_dir / "_kernel_log.txt"
        log_path.write_text(plain)
        print(f"  - {log_path.name} ({log_path.stat().st_size} bytes, kernel log)")


def push_and_wait(
    folder: Path,
    output_dir: Path,
    poll_interval: int = DEFAULT_POLL_INTERVAL,
    timeout: int = DEFAULT_TIMEOUT,
) -> int:
    print("Pushing notebook...")
    pushed = push(folder)
    slug = pushed["_slug"]
    version = pushed.get("versionNumber") or pushed.get("version_number") or "?"
    url = pushed.get("url") or f"https://www.kaggle.com/code/{slug}"
    print(f"  pushed version {version}: {url}")

    if pushed.get("error"):
        print(f"  push reported error: {pushed['error']}")
        return 2

    print("Polling status (Ctrl-C to detach)...")
    deadline = time.time() + timeout
    last = None
    terminal = {"complete", "error", "cancelled", "cancelAcknowledged", "stopped"}
    while time.time() < deadline:
        s = status(slug)
        st = (s.get("status") or s.get("Status") or "").lower()
        msg = s.get("failureMessage") or s.get("FailureMessage") or ""
        line = f"  [{time.strftime('%H:%M:%S')}] status={st or '?'}"
        if msg:
            line += f"  msg={msg[:200]}"
        if line != last:
            print(line)
            last = line
        if st in terminal:
            print(f"\nDownloading outputs to {output_dir} ...")
            download_output(slug, output_dir)
            return 0 if st == "complete" else 1
        time.sleep(poll_interval)

    print(f"Timed out after {timeout}s; kernel may still be running.")
    return 3


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("push")
    sp.add_argument("--folder", type=Path, default=Path("notebooks"))

    ss = sub.add_parser("status")
    ss.add_argument("--slug", type=str, required=True)

    so = sub.add_parser("output")
    so.add_argument("--slug", type=str, required=True)
    so.add_argument("--output-dir", type=Path, default=Path("outputs/kaggle_run"))

    sw = sub.add_parser("push-and-wait")
    sw.add_argument("--folder", type=Path, default=Path("notebooks"))
    sw.add_argument("--output-dir", type=Path, default=Path("outputs/kaggle_run"))
    sw.add_argument("--poll-interval", type=int, default=DEFAULT_POLL_INTERVAL)
    sw.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)

    args = p.parse_args(argv)
    if args.cmd == "push":
        print(json.dumps(push(args.folder), indent=2))
        return 0
    if args.cmd == "status":
        print(json.dumps(status(args.slug), indent=2))
        return 0
    if args.cmd == "output":
        download_output(args.slug, args.output_dir)
        return 0
    if args.cmd == "push-and-wait":
        return push_and_wait(
            args.folder, args.output_dir, args.poll_interval, args.timeout
        )
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
