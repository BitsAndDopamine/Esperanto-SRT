"""Command line interface: transcribe audio files to .srt subtitles.

Usage:
    eo-stt-cli episode.mp3 [more.mp3 ...] [-o OUTPUT_DIR] [--model MODEL_ID] [--device cpu|cuda|auto]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .model import MODEL_ID, load_model
from .transcribe import transcribe_file


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        prog="eo-stt-cli",
        description="Transcribe Esperanto audio (mp3/wav/ogg/flac) to .srt subtitles.",
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="audio files to transcribe")
    parser.add_argument("-o", "--output-dir", type=Path, default=None,
                        help="where to write .srt files (default: next to each input)")
    parser.add_argument("--model", default=MODEL_ID, help=f"NeMo model id (default: {MODEL_ID})")
    parser.add_argument("--device", default="auto", choices=["cpu", "cuda", "auto"],
                        help="inference device (default: auto)")
    args = parser.parse_args(argv)

    missing = [p for p in args.inputs if not p.is_file()]
    if missing:
        parser.error(f"no such file(s): {', '.join(str(p) for p in missing)}")

    print(f"Loading model {args.model} (first run downloads ~500 MB)...", flush=True)
    model = load_model(args.model, device=args.device)
    for path in args.inputs:
        print(f"Transcribing {path} ...", flush=True)

        def _progress(done: int, total: int) -> None:
            print(f"\r  {done}/{total} chunks ({done * 100 // total}%)", end="", flush=True)

        out = transcribe_file(model, path, args.output_dir, on_progress=_progress)
        print(f"\r  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
