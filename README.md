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

Writes `episode.srt` next to each input (or into the `-o` directory). Accepted input formats: mp3, wav, ogg, flac, m4a.

## How it works

1. Audio is decoded and resampled to 16 kHz mono (`eo_stt/audio.py`).
2. Long files are transcribed in 30 s chunks with NeMo (`eo_stt/model.py`).
3. The transcript is split into timed subtitle blocks and written as srt (`eo_stt/srt.py`, `eo_stt/transcribe.py`).

Note: the model outputs **lowercase text without punctuation** — the srt text is passed through as-is.

## Development

```bash
.venv/bin/python -m pytest
```

The model checkpoint (~500 MB) is downloaded on first run into the Hugging Face cache (`~/.cache/huggingface` on Linux) and is never committed.
