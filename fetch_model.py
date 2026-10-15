"""Download the INT8 ONNX ASR model into models/ (build-time only, never at runtime).

    python fetch_model.py            # download if missing
    python fetch_model.py --force    # re-download

Fetches ONLY the INT8 model files (plus config.json and vocab.txt) from the
pinned Hugging Face commit of OpenVoiceOS/nvidia-eo-conformer-transducer-large-onnx.
The fp32 variants are NOT downloaded. Every file is verified against its
pinned SHA-256, so a corrupted or tampered download fails loudly.

The app itself never downloads anything: it loads the model exclusively from
the local directory produced here (bundled into the PyInstaller build).
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "nvidia-eo-conformer-transducer-large-onnx-int8"

REPO_ID = "OpenVoiceOS/nvidia-eo-conformer-transducer-large-onnx"
# Pinned commit of the HF repo (2026-07-04).
REVISION = "11e3daa853020aaa2b9dfc1a2a6b9983c6f89802"

# INT8 files only (the fp32 encoder/decoder + .onnx_data are intentionally absent).
# sha256 values are the repo's LFS oids for the model files; the two small text
# files are pinned by their content hash as well.
FILES: dict[str, tuple[str, int]] = {
    "encoder-model.int8.onnx": (
        "8eff0d620550ae3a3390c1f63c6cb554f51852bbdba0240f4bd315a2e1a7cbe7",
        137_508_212,
    ),
    "decoder_joint-model.int8.onnx": (
        "f5222232acdbba4d5bb1ff3f8df20b3b4b2944f49314917f3ff0a79cc9828e17",
        4_214_216,
    ),
    "config.json": ("56853d83afabf9bf4b7a4f0f0e262d49a0d192757dee9214eb67b9bb239df59c", 120),
    "vocab.txt": ("acf7c76ccb017bf1f4bdfc9b8eca018433a7e43fe75fe98cd49796d46af25fea", 907),
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _download(name: str, dest: Path) -> None:
    url = f"https://huggingface.co/{REPO_ID}/resolve/{REVISION}/{name}"
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  {name} <- {url}", flush=True)
    with urllib.request.urlopen(url) as resp, tmp.open("wb") as out:
        shutil.copyfileobj(resp, out, length=1 << 20)
    tmp.replace(dest)


def fetch_model(force: bool = False) -> Path:
    """Ensure the pinned INT8 model files exist locally. Returns the model dir."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name, (sha256, size) in FILES.items():
        dest = MODEL_DIR / name
        if dest.is_file() and not force:
            actual = _sha256(dest)
            if actual == sha256 and dest.stat().st_size == size:
                print(f"  {name}: OK (cached)")
                continue
            print(f"  {name}: checksum mismatch, re-downloading")
        _download(name, dest)
        actual = _sha256(dest)
        if actual != sha256 or dest.stat().st_size != size:
            dest.unlink(missing_ok=True)
            raise RuntimeError(
                f"checksum mismatch for {name}: expected {sha256}, got {actual} "
                f"(size {dest.stat().st_size if dest.exists() else 'missing'} != {size})"
            )
        print(f"  {name}: OK ({size / 1e6:.1f} MB)")
    return MODEL_DIR


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="re-download even if files exist")
    args = ap.parse_args()
    print(f"Fetching INT8 model {REPO_ID} @ {REVISION} -> {MODEL_DIR}")
    model_dir = fetch_model(force=args.force)
    total = sum((model_dir / n).stat().st_size for n in FILES)
    print(f"Model ready: {model_dir} ({total / 1e6:.1f} MB total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
