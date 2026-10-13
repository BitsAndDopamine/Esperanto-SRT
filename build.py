"""Build the portable executable with PyInstaller (one folder per OS).

    .venv/bin/python build.py             # onedir build for THIS OS -> dist/eo-stt/
    .venv/bin/python build.py --onefile   # single file (slow startup with a multi-GB bundle)

PyInstaller cannot cross-compile: run this once per target OS (Linux, Windows,
macOS). .github/workflows/build.yml does exactly that in CI.

Notes:
- The venv uses CPU-only torch, so the bundle has no CUDA libraries.
- The ASR model (~500 MB) is NOT bundled; the app downloads it to the
  Hugging Face cache on first run.
- Expect a large bundle (~1.5-3 GB): PyTorch + NeMo are big.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Packages whose data/submodules/native libs PyInstaller's static analysis misses.
COLLECT_ALL = [
    "customtkinter",   # themes, fonts, assets
    "nemo",            # no official PyInstaller hook; yaml configs + submodules
    "lhotse",
    "soundfile",       # libsndfile binary
    "miniaudio",       # native decoder
    "sentencepiece",
    "tokenizers",
    "safetensors",
    "omegaconf",
    "hydra",
    "huggingface_hub",
    "cuda",            # cuda.bindings.* compiled modules (nemo cuda utils import them)
    # these read a package-local version.info file at import time (not metadata)
    "lightning_fabric",
    "pytorch_lightning",
]

COPY_METADATA = ["nemo_toolkit", "huggingface_hub", "tokenizers", "safetensors", "regex"]

# Not used by the ASR path (verified: not in sys.modules after importing it).
# Do NOT exclude stdlib test modules (unittest/pydoc/doctest): numpy.testing
# imports unittest and the frozen app crashed without it.
EXCLUDE = [
    "nemo.collections.tts",
    "nemo.collections.nlp",
    "nemo.collections.vision",
    "matplotlib",
    "IPython",
    "jupyter",
    "tkinter.test",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--onefile", action="store_true", help="single-file build (slow startup)")
    ap.add_argument("--name", default="eo-stt")
    args = ap.parse_args()

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm", "--clean",
        "--name", args.name,
        "--distpath", str(ROOT / "dist"),
        "--workpath", str(ROOT / "build"),
        "--paths", str(ROOT),
        "--onefile" if args.onefile else "--onedir",
        str(ROOT / "main.py"),
    ]
    for pkg in COLLECT_ALL:
        cmd += ["--collect-all", pkg]
    for pkg in COPY_METADATA:
        cmd += ["--copy-metadata", pkg]
    for mod in EXCLUDE:
        cmd += ["--exclude-module", mod]

    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    print(f"\nBuild finished -> {ROOT / 'dist' / args.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
