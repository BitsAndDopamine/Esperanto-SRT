# AGENTS.md — eo-stt

## What this project is

Esperanto speech-to-text desktop app: a **graphical interface** that converts **mp3 audio files into .srt subtitle files**, running on **Linux and Windows**.

Stack (decided): **Python** + NVIDIA NeMo + PyTorch, **Tkinter** GUI (stdlib, zero-dep, cross-platform). CLI + GUI share one pipeline (`eo_stt/`).

## The model (decided — don't swap without discussion)

- NVIDIA NeMo **`stt_eo_conformer_ctc_large`** — Conformer-CTC large (~120M params), Esperanto.
  HF mirror (used in code): `nvidia/stt_eo_conformer_ctc_large` (license CC-BY-4.0).
  NGC catalog: https://catalog.ngc.nvidia.com/orgs/nvidia/nemo/models/stt_eo_conformer_ctc_large
- Load: `nemo_asr.models.EncDecCTCModelBPE.from_pretrained("nvidia/stt_eo_conformer_ctc_large")` — the **BPE** variant (SentencePiece, vocab 128), not plain `EncDecCTCModel`.
- **Input: 16 kHz mono.** mp3/wav/ogg/flac decoding + resampling lives in `eo_stt/audio.py` (soundfile + scipy).
- **Output: lowercase Esperanto text with spaces and apostrophes — no capitalization, no punctuation.** The srt text is passed through as-is (decided: no post-processing).
- **Word timestamps**: `model.transcribe(paths, timestamps=True)` returns Hypothesis objects with `.timestamp['word']` = list of `{"word", "start", "end"}` (seconds; NeMo converts frames→seconds itself). Used for srt blocks in `eo_stt/srt.py::segments_from_words`. Fallback if unavailable: `proportional_segments`.
- Long audio is transcribed in **30 s chunks** (`CHUNK_SECONDS` in `eo_stt/model.py`) — bounded memory on CPU.
- The `.nemo` checkpoint (~500 MB) downloads on first run into the Hugging Face cache (`~/.cache/huggingface`). **Never commit it** (`.gitignore` covers `*.nemo`).

## Environment / setup (verified on this machine)

- **Python must be 3.10–3.12** — NeMo does not support 3.13/3.14 (the system python here is 3.14). Use uv or a python3.12 venv:
  ```bash
  uv venv --python 3.12 .venv
  uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu  # CPU-only wheel; drop the index for CUDA
  uv pip install --python .venv/bin/python -e ".[test]"
  ```
- **No ffmpeg needed** — `soundfile` (libsndfile ≥ 1.1) decodes mp3/wav/ogg/flac directly (covered by `tests/test_audio.py::test_load_audio_mp3`). ffmpeg is only a fallback path in `eo_stt/audio.py` when soundfile fails.
- This machine has **no GPU and no sudo** — CPU inference only; tests and e2e run on CPU.

## Commands

```bash
.venv/bin/python -m pytest                 # tests (22, no model download needed)
.venv/bin/python -m eo_stt                 # GUI (needs a display)
.venv/bin/eo-stt-cli FILE.mp3 [...] -o OUT_DIR   # CLI, one .srt per input
```

Output `.srt` goes next to the input (same stem) or into `-o` dir. Accepted inputs: mp3, wav, ogg, flac, m4a.

## Layout

- `eo_stt/audio.py` — decode any format → 16 kHz mono float32 (soundfile + scipy resample; ffmpeg fallback).
- `eo_stt/model.py` — NeMo load + chunked transcription with word timestamps (lazy heavy imports).
- `eo_stt/srt.py` — pure srt logic: `segments_from_words` (real timestamps), `proportional_segments` (fallback), `write_srt`. Block rules: ≤84 chars (2×42), ≤7 s, break on >1.5 s gaps, min 0.5 s per block.
- `eo_stt/transcribe.py` — file → segments → .srt orchestration.
- `eo_stt/cli.py` / `eo_stt/gui.py` — CLI and Tkinter GUI (worker thread + event queue; GUI logic stays thin).
- `tests/` — pytest; model-independent (StubModel), incl. a synthetic mp3 encode/decode test.

## Gotchas

- NeMo import name is **`nemo`**, not `nemo_toolkit` (pip package vs import name).
- CTC decoding uses **`greedy_batch`** (set in `load_model`) — NeMo's recommended strategy, same output as `greedy` but faster; verified to keep word timestamps working.
- `lameenc` API is `lameenc.Encoder()` (set_bit_rate/set_in_sample_rate/…), there is no `lameenc.encode()`.
- Model output is lowercase without punctuation — don't "fix" it in code; it's the model's raw output.
- Hard 30 s chunking has no overlap: a word straddling a chunk boundary can come out garbled/duplicated (seen e2e as `esperantoo`). Acceptable tradeoff; adding overlap+dedup is the fix if it bothers users.
- The GUI has **not been run on a display** on this machine (headless, no X) — only import-tested. The CLI shares the same `transcribe_file` path, which is e2e-verified.
- GUI must stay cross-platform: no shell-outs, no Linux-only paths; all file dialogs via tkinter.

## Verified end-to-end (Oct 2026, CPU)

Real Esperanto speech (Wikimedia Commons, CC BY-SA 4.0, "Mi estas komencanto de esperanto", 3.3 s mp3) → `eo-stt-cli sentence.mp3` → correct transcript `mi estas komencanto de esperanto` with word-aligned timing (0.6–2.88 s). A 44 s file exercised the 2-chunk path (6 srt blocks, chunk boundary at 30 s respected). ~44 s of audio transcribes in ~5 s of inference on this CPU.

## Workspace context

- `../vocx` (Go) is the inverse tool: Esperanto text → phonetic Polish for TTS engines. Same domain (Esperanto audio tooling), independent stack — not a dependency.
