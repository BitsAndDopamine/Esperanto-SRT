"""Command line interface: transcribe audio files to .srt subtitles.

Usage:
    eo-stt-cli episode.mp3 [more.mp3 ...] [-o OUTPUT_DIR] [--model MODEL_DIR]

Inference is CPU-only ONNX Runtime (INT8 model, loaded from local files;
nothing is downloaded at runtime).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .model import default_model_dir, load_model
from .transcribe import transcribe_file


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        prog="eo-stt-cli",
        description="transskribas esperantan parolon (mp3/wav/ogg/flac) al .srt-subtitoloj.",
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="dosieroj por transskribi")
    parser.add_argument("-o", "--output-dir", type=Path, default=None,
                        help="kie skribi .srt-dosierojn (apriore: apud cxiu enigo)")
    parser.add_argument("--model", default=None, metavar="MODEL_DIR",
                        help="vojo al la loka INT8-modelodosierujo (apriore: la pakita modelo)")
    parser.add_argument("--device", default="auto", choices=["cpu", "cuda", "auto"],
                        help="ignorata: inferenco nur per CPU (ONNX Runtime)")
    args = parser.parse_args(argv)

    missing = [p for p in args.inputs if not p.is_file()]
    if missing:
        parser.error(f"ne ekzistas dosiero(j): {', '.join(str(p) for p in missing)}")

    model_dir = Path(args.model) if args.model else default_model_dir()
    print(f"preparante modelon de {model_dir} ...", flush=True)
    model = load_model(model_dir)
    for path in args.inputs:
        print(f"transskribante {path} ...", flush=True)

        def _progress(done: int, total: int) -> None:
            print(f"\r  {done}/{total} pecoj ({done * 100 // total}%)", end="", flush=True)

        out = transcribe_file(model, path, args.output_dir, on_progress=_progress)
        print(f"\r  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
