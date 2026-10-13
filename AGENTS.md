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
- **Word timestamps**: `model.transcribe(arrays, timestamps=True)` returns Hypothesis objects with `.timestamp['word']` = list of `{"word", "start", "end"}` (seconds; NeMo converts frames→seconds itself). Used for srt blocks in `eo_stt/srt.py::segments_from_words`. Fallback if unavailable: `proportional_segments`.
- Long audio is transcribed in **30 s chunks** (`CHUNK_SECONDS` in `eo_stt/model.py`), **4 chunks per `model.transcribe` call** (`BATCH_CHUNKS`) — bounded memory on CPU, amortized dataloader setup.
- **Pass audio to NeMo as numpy arrays, never as file paths.** The file-path route goes through a Lhotse manifest dataloader that spawns worker processes per call — that broke in the GUI's worker thread (root cause of the "only the first ~56 s transcribed" bug: chunks after the first came back empty). The tensor route (`_transcribe_arrays` in `eo_stt/model.py`) is in-process (`num_workers=0`), no temp files, no Lhotse.
- If a batched call fails, `transcribe_chunks` retries that batch one chunk per call (isolation).
- Progress: `on_progress(done, total)` callback → CLI prints chunk %, GUI progress bar reflects chunk progress within each file.
- `transcribe_to_segments` logs a **warning if the srt would cover < 50% of a > 60 s file** (catches partial-failure runs like the one above).
- Speed on CPU: ~11× realtime → a 48 min podcast takes ~4–5 min (verified on fremdulo.mp3); GPU is much faster.
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
.venv/bin/python -m pytest                 # tests (27, no model download needed)
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
- **Don't route audio through NeMo's file-path/Lhotse path** (see above) — it silently returned empty transcripts for all but the first ~2 chunks when run from the GUI thread. If a future NeMo upgrade tempts you to pass paths again, re-test the GUI on a > 5 min file first.
- **All GUI/CLI strings must be ASCII** — enforced by `tests/test_ui_strings.py`. Tk's default font here is Latin-1-only "Nimbus Sans L", which garbles `…`, `→`, `—` and the Esperanto diacritics (ĉĝĥĵŝŭ) in *widget* text (window titles are WM-rendered and looked fine — misleading). `_setup_fonts()` in `gui.py` switches widget fonts to a Unicode-capable family (DejaVu Sans first) so Esperanto *file names* render; app strings stay ASCII regardless. Windows consoles can also fail to encode non-ASCII prints — another reason the CLI is ASCII-only.
- The GUI **has been visually verified on this machine's display** (screenshots: buttons, listbox with a diacritic file name `ĉeĥa-ĝojo ŝido.mp3`). The transcription button itself was not clicked in the GUI, but the worker-thread code path is tested headlessly with the real model, and the CLI shares the same `transcribe_file` path (e2e-verified).
- GUI must stay cross-platform: no Linux-only paths. File dialogs use the **OS-native selector**: `zenity` (the native GTK/GNOME dialog) on Linux when installed, `tkinter.filedialog` elsewhere (already native on Windows/macOS; Tk dialog on Linux without zenity). Helpers `_ask_open_filenames`/`_ask_directory` in `gui.py` (unit-tested with mocks in `tests/test_gui.py`). m4a/aac is NOT in the GUI's file picker (no bundled decoder; needs ffmpeg fallback).

## Verified end-to-end (Oct 2026, CPU)

- Real Esperanto speech (Wikimedia Commons, CC BY-SA 4.0, "Mi estas komencanto de esperanto", 3.3 s mp3) → correct transcript with word-aligned timing (0.6–2.88 s).
- **Full 48-min podcast** (`fremdulo.mp3`, user's file, 2894.6 s) → `eo-stt-cli` → 498 srt blocks, coverage 10.6 s → 2894.5 s (full file), only 4 natural silence gaps. Runs in ~5 min on CPU.
- The same file via the **old file-path/Lhotse route in a GUI worker thread** produced only the first ~56 s (8 blocks) — reproduced the user's bug; the tensor-route fix eliminates it (thread path re-verified headlessly with the real model).
- Accepted inputs tested: mp3 (CBR/VBR, 44.1 kHz mono), wav, ogg.

## Portable builds (PyInstaller)

- `build.py` builds a standalone app for **this OS only** (PyInstaller cannot cross-compile): `.venv/bin/python build.py` → `dist/eo-stt/` (onedir, ~1.5-3 GB — torch+NeMo are big; CPU-only torch, no CUDA).
- The exe: **no args → GUI; with args → CLI** (`eo-stt FILE.mp3 -o OUT_DIR`). Entry point is root `main.py` (kept outside the package for PyInstaller).
- `.github/workflows/build.yml` builds Linux/Windows/macOS zips on `v*` tags and attaches them to a GitHub release. CI uses python.org Python (bundles Tk 8.6 — see font note).
- The model (~500 MB) is **never bundled**; first run downloads it to the HF cache.
- PyInstaller flags that matter (in `build.py`): `--collect-all` for customtkinter/nemo/lhotse/soundfile/miniaudio/sentencepiece/tokenizers/safetensors/omegaconf/hydra/huggingface_hub (assets + natives + no official hooks), `--copy-metadata` for nemo_toolkit etc., excludes for unused nemo collections (tts/nlp/vision — verified not imported by the ASR path; pandas IS imported, keep it).
- **Frozen-app startup crashes fixed (verified)**: (1) do NOT `--exclude-module unittest`/`pydoc` — `numpy.testing` imports unittest and the frozen app died without it; (2) `--collect-all cuda` — nemo's cuda utils import `cuda.bindings` compiled modules; (3) `--collect-all lightning_fabric` + `pytorch_lightning` — their `__version__.py` reads a package-local `version.info` file at import time.
- Verified build (Linux, Oct 2026): `dist/eo-stt/` = **1.5 GB** (exe 95 MB + `_internal/` 1.4 GB). Frozen CLI transcribes correctly (sentence.mp3 → right srt in ~15 s incl. startup; long.mp3 → 2/2 chunks); frozen GUI launches (dark CTk window).
- GUI is **CustomTkinter** (dark mode). Caveats on this machine's Linux/Xwayland + Tk: (1) Tk named fonts only resolve core X11 fonts, so CTk's TrueType fonts render as the bitmap "fixed" font (pixelated text; diacritics still show) — cosmetic only, user said fonts are not a priority; (2) **set `corner_radius=0` on all CTk widgets** — CTk's rounded corners render as jagged notches here (broken-looking buttons); sharp rectangles render cleanly. On Windows/macOS (native font/canvas backends) CTk renders properly.

## Workspace context

- `../vocx` (Go) is the inverse tool: Esperanto text → phonetic Polish for TTS engines. Same domain (Esperanto audio tooling), independent stack — not a dependency.
