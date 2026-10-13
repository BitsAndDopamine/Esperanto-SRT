# eo-stt

Esperanto speech-to-text: turn mp3 audio into `.srt` subtitle files, with a small desktop GUI. Runs on **Linux and Windows** (CPU or NVIDIA GPU).

Uses NVIDIA's NeMo model [`stt_eo_conformer_ctc_large`](https://catalog.ngc.nvidia.com/orgs/nvidia/nemo/models/stt_eo_conformer_ctc_large) (Conformer-CTC, ~120M params, trained on Mozilla Common Voice Esperanto, license CC-BY-4.0).

## Install

Requires **Python 3.10–3.12** (NeMo does not support 3.13/3.14).

```bash
python3.12 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu  # CPU-only; drop the index URL for CUDA
.venv/bin/pip install -e ".[test]"
```

On Windows use `.venv\Scripts\pip.exe …` and `python` instead of `python3.12`.

No ffmpeg needed: mp3/wav/ogg/flac decoding is handled by `soundfile` (libsndfile ≥ 1.1 reads mp3 directly).

## Run

GUI:

```bash
.venv/bin/python -m eo_stt
```

CLI:

```bash
.venv/bin/eo-stt-cli episode.mp3 more.mp3 -o subtitles/
```

Writes `episode.srt` next to each input (or into the `-o` directory). Accepted input formats: mp3, wav, ogg, flac (m4a/aac only when ffmpeg is installed). Both the GUI and the CLI show chunk-level progress, so long files don't look stuck.

On CPU, inference runs at roughly 10× realtime (a 48-minute podcast takes ~4–5 minutes); with a CUDA GPU it is much faster.

## How it works

1. Audio is decoded and resampled to 16 kHz mono (`eo_stt/audio.py`).
2. Long files are transcribed in 30 s chunks with NeMo, 4 chunks per inference batch (`eo_stt/model.py`). Audio is passed to NeMo as numpy arrays (in-process, no worker processes).
3. The transcript is split into timed subtitle blocks and written as srt (`eo_stt/srt.py`, `eo_stt/transcribe.py`).

Note: the model outputs **lowercase text without punctuation** — the srt text is passed through as-is.

## Portable builds (standalone executable)

Build a self-contained app with PyInstaller — no Python needed on the target machine:

```bash
.venv/bin/python build.py          # -> dist/eo-stt/ (folder, ~1.5-3 GB)
```

- The executable with **no arguments opens the GUI**; with arguments it works as the CLI: `eo-stt fremdulo.mp3 -o out/`.
- PyInstaller **cannot cross-compile** — build once per OS. CI builds Linux, Windows and macOS artifacts on every `v*` tag (`.github/workflows/build.yml`).
- The bundle uses **CPU-only PyTorch** (no CUDA); inference is ~10x realtime on a CPU.
- The ASR model (~500 MB) is **not bundled** — it downloads to the Hugging Face cache on first run (`~/.cache/huggingface`; `%USERPROFILE%\.cache\huggingface` on Windows). To run fully offline, run the app once with internet access, or copy `stt_eo_conformer_ctc_large.nemo` into that cache.
- Prefer the default **onedir** build; `--onefile` re-extracts the whole bundle on every start (slow).

## Development

```bash
.venv/bin/python -m pytest
```

The model checkpoint (~500 MB) is downloaded on first run into the Hugging Face cache (`~/.cache/huggingface` on Linux) and is never committed.
