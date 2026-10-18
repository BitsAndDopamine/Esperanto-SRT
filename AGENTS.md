# AGENTS.md — eo-stt

## What this project is

Esperanto speech-to-text desktop app: a **graphical interface** that converts **mp3 audio files into .srt subtitle files**, running on **Linux and Windows**.

Stack: **Python** + ONNX Runtime (CPU) + onnx-asr, **Tkinter/CustomTkinter** GUI (stdlib Tk + customtkinter, cross-platform). CLI + GUI share one pipeline (`eo_stt/`). **No PyTorch, NeMo, CUDA or scipy** — the whole inference stack is onnxruntime + onnx-asr + numpy, keeping the packaged app at ~250-350 MB instead of 1.5+ GB.

## The model (decided — don't swap without discussion)

- INT8 ONNX export of NVIDIA's Esperanto **Conformer-Transducer Large** (~120M params): HF repo `OpenVoiceOS/nvidia-eo-conformer-transducer-large-onnx` (license CC-BY-4.0), converted from `nvidia/stt_eo_conformer_transducer_large`.
- **License**: the **code** is CC0 (`LICENSE`); the bundled **model** files stay CC-BY-4.0 (HF repo above) — keep the model's attribution in the README when redistributing.
- **INT8 files only**: `encoder-model.int8.onnx` (~131 MB), `decoder_joint-model.int8.onnx` (~4 MB), plus `config.json` and `vocab.txt`. Total ~142 MB. The fp32 variants are never downloaded.
- **Loaded locally, always.** `fetch_model.py` downloads the pinned HF commit (`11e3daa…`, sha256-verified) into `models/nvidia-eo-conformer-transducer-large-onnx-int8/` at BUILD time. The app never downloads anything at runtime. `eo_stt/model.py::default_model_dir()` resolves the model dir: `$EO_STT_MODEL_DIR` → frozen: `<exe dir>/model` then `_MEIPASS/model` → source tree: `<repo>/models/<name>`.
- **Backend: onnx-asr** (`onnx_asr.load_model(MODEL_ID, path=..., quantization="int8", providers=["CPUExecutionProvider"])`) + **onnxruntime CPU** (plain `onnxruntime` wheel, never `onnxruntime-gpu`). CPU-only: the `--device` CLI flag is accepted but ignored.
- **Input: 16 kHz mono float32 numpy arrays** passed to `recognize()` — never file paths (onnx-asr reads wav paths with the stdlib `wave` module only; mp3 needs our soundfile decode anyway).
- **Output: lowercase Esperanto text with diacritics (ĉĝĥĵŝŭ), no punctuation.** The srt text is passed through as-is (decided: no post-processing).
- **Word timestamps**: onnx-asr's `with_timestamps().recognize(arrays)` returns token-level timestamps (0.01 s hop × 4 subsampling = 0.04 s/frame). `eo_stt/model.py::_words_from_tokens` regroups SentencePiece tokens (▁ → space) into `(word, start, end)` tuples; token i spans `[t_i, t_{i+1})`, last token +1 frame. Exposed via hypothesis objects mimicking NeMo's `.text` / `.timestamp['word']`, so `srt.py::segments_from_words` is unchanged. Fallback if timestamps are unavailable: `proportional_segments`.
- Long audio is transcribed in **30 s chunks** (`CHUNK_SECONDS` in `eo_stt/model.py`), **4 chunks per `recognize` call** (`BATCH_CHUNKS`) — bounded memory on CPU, amortized feature extraction. If a batched call fails, `transcribe_chunks` retries that batch one chunk per call (isolation).
- Progress: `on_progress(done, total)` callback → CLI prints chunk %, GUI progress bar reflects chunk progress within each file.
- `transcribe_to_segments` logs a **warning if the srt would cover < 50% of a > 60 s file** (catches partial-failure runs).
- Speed on CPU: ~13× realtime → a 48 min podcast takes ~3-4 min (verified on fremdulo.mp3 with the INT8 model).
- **Offline guarantee**: onnx-asr only imports `huggingface_hub` lazily when downloading; with a local `path` that exists it resolves purely from disk. `huggingface_hub` is not even a dependency of onnx-asr 0.12 (only numpy is).

## Environment / setup (verified on this machine)

- **Python 3.10+** (onnx-asr and onnxruntime both support 3.10–3.14; no NeMo constraint anymore).
  ```bash
  uv venv --python 3.12 .venv
  uv pip install --python .venv/bin/python -e ".[test,build]"
  python fetch_model.py        # one-time: download the INT8 model into models/
  ```
- **No ffmpeg needed** — `soundfile` (libsndfile ≥ 1.1) decodes mp3/wav/ogg/flac directly (covered by `tests/test_audio.py::test_load_audio_mp3`). ffmpeg is only a fallback path in `eo_stt/audio.py` when soundfile fails.
- **No scipy** — resampling is a numpy-only polyphase resampler (`_resample_poly` in `eo_stt/audio.py`, same Kaiser-windowed-sinc design as `scipy.signal.resample_poly`, runs in blocks for bounded memory). scipy cost ~100 MB installed; not worth it for one function.
- This machine has **no GPU and no sudo** — CPU inference only; tests and e2e run on CPU.

## Commands

```bash
.venv/bin/python -m pytest                 # tests (45; integration test uses the local model)
.venv/bin/python -m eo_stt                 # GUI (needs a display)
.venv/bin/eo-stt-cli FILE.mp3 [...] -o OUT_DIR   # CLI, one .srt per input
python fetch_model.py                      # download/verify the INT8 model into models/
.venv/bin/python build.py                  # PyInstaller onedir build -> dist/eo-stt/
```

Output `.srt` goes next to the input (same stem) or into `-o` dir. Accepted inputs: mp3, wav, ogg, flac, m4a (m4a via the ffmpeg fallback only).

## Layout

- `eo_stt/audio.py` — decode any format → 16 kHz mono float32 (soundfile; numpy polyphase resample; ffmpeg fallback).
- `eo_stt/model.py` — onnx-asr/ONNX Runtime load + chunked transcription with word timestamps (lazy heavy imports).
- `eo_stt/srt.py` — pure srt logic: `segments_from_words` (real timestamps), `proportional_segments` (fallback), `write_srt` (all at once) and `SrtWriter` (incremental). Block rules: ≤84 chars (2×42), ≤7 s, break on >1.5 s gaps, min 0.5 s per block.
- `eo_stt/transcribe.py` — file → segments → .srt orchestration. The .srt is written incrementally (`SrtWriter`): created up front, appended per chunk (via `transcribe_to_segments`' `on_segments` callback; `transcribe_chunks` is a generator that yields per chunk), so the file can be read while transcription runs. A failed run keeps the partial file; a completely empty transcript raises ValueError and leaves no file.
- `eo_stt/cli.py` / `eo_stt/gui.py` — CLI and CustomTkinter GUI (worker thread + event queue; GUI logic stays thin).
- `fetch_model.py` — build-time model fetch (pinned commit + sha256 per file) into `models/` (gitignored).
- `build.py` — generates the PyInstaller spec, builds onedir, renames the executable to carry the version (`eo-stt-v<version>-linux` on Linux, `eo-stt-v<version>.exe` on Windows; from `--version` or `v` + `eo_stt.__version__`), and copies the model to `dist/eo-stt/model/`. The version's single source is `eo_stt/__init__.py::__version__` (pyproject.toml reads it dynamically; the GUI's "pri" window shows it too).
- `tests/` — pytest; model-independent (StubModel for the pipeline), plus a real-model integration test that skips when `models/` is absent, incl. a synthetic mp3 encode/decode test.

## Gotchas

- Model output is lowercase without punctuation — don't "fix" it in code; it's the model's raw output.
- Hard 30 s chunking has no overlap: a word straddling a chunk boundary can come out garbled/duplicated. Acceptable tradeoff; adding overlap+dedup is the fix if it bothers users.
- **Interface strings are Esperanto with diacritics (ĉĝĥĵŝŭ) in the GUI; the CLI stays ASCII.** `tests/test_ui_strings.py` enforces: gui.py may contain nothing beyond the Esperanto diacritics outside ASCII, cli.py must be pure ASCII (Windows consoles can fail to encode non-ASCII prints). Tk's default font here is Latin-1-only "Nimbus Sans L", which would garble the diacritics in *widget* text (window titles are WM-rendered and looked fine — misleading). `_default_font_family()`/`_setup_tk_fonts()` in `gui.py` switch CTk widgets (theme font) and Tk's named fonts (TkDefaultFont/TkTextFont/TkMenuFont — used by messagebox and the Tk file-dialog fallback) to a Unicode-capable family (DejaVu Sans first) so the diacritics and Esperanto *file names* render. (Transcript text is Esperanto and lives only in the UTF-8 .srt files.)
- The GUI has a **preview window** (`PreviewWindow` in `gui.py`): a read-only view of a finished .srt with a "kopii ĉion" (copy-all) button. It opens automatically for the last file when a run completes; each file row also has an "srt" button to preview that file's .srt on demand — it is **enabled only while the .srt exists** (it lights up as soon as a file's transcription starts, since the .srt grows incrementally; `_refresh_srt_buttons` re-checks on worker events and when the output folder changes). A **"pri"** button opens an about window (`AboutWindow`) with the version, the GitHub repo and the licenses (code CC0 1.0, model CC-BY-4.0).
- The GUI **has been visually verified on this machine's display**, including the preview window and diacritic messageboxes (screenshots). The transcription button itself was not clicked in the GUI, but the worker-thread code path (including the auto-opened preview and the copy-to-clipboard button) is tested headlessly with the real model, and the CLI shares the same `transcribe_file` path (e2e-verified).
- The GUI has a **status line under the progress bar**. Clicking "transskribu" immediately shows activity while the model loads (busy marquee animation on the bar via `CTkProgressBar.start()`, status "ŝargante modelon..."); model loading then takes the first 5% of the bar (`MODEL_LOAD_SHARE` in `gui.py`), the files share the remaining 95%, and the status shows the current file ("transskribas [i/n] ...", "farita.", "eraro"). The busy animation is always stopped when the run ends. The worker-thread progress/status flow is covered by display-gated tests in `tests/test_gui.py` (skipped on headless CI) and was e2e-verified with the real model.
- The file-open dialog's **type dropdown lists each audio format separately** (`mp3-dosieroj (*.mp3)`, `wav-dosieroj (*.wav)`, ... in addition to the preselected "aŭdiaj dosieroj" and "ĉiuj dosieroj"): zenity gets one `--file-filter` per format, the Tk fallback gets the same list as `filetypes`. Some native dialogs mishandle one entry with several patterns, which is why the combined entry alone did nothing.
- GUI must stay cross-platform: no Linux-only paths. File dialogs use the **OS-native selector**: `zenity` (the native GTK/GNOME dialog) on Linux when installed, `tkinter.filedialog` elsewhere (already native on Windows/macOS; Tk dialog on Linux without zenity). Helpers `_ask_open_filenames`/`_ask_directory` in `gui.py` (unit-tested with mocks in `tests/test_gui.py`). m4a/aac is NOT in the GUI's file picker (no bundled decoder; needs ffmpeg fallback).
- **Frozen-app subprocess env (v0.1.0 Linux bug, fixed)**: the PyInstaller bootloader points `LD_LIBRARY_PATH` at the bundled `_internal/` (a `setenv` after startup — invisible in `/proc/<pid>/environ`, but inherited by children). Children must not inherit it: the CI build (python.org Python, Tk 8.6) bundles system libs incl. an older `libfontconfig`, and the `zenity` child died with `symbol lookup error: libpangoft2-1.0.so.0: undefined symbol: FcConfigSetDefaultSubstitute` — so the v0.1.0 Linux artifact's file dialogs silently did nothing (any non-zero zenity exit was read as "cancelled"). A local build with uv's Python (Tk 9, almost no bundled system libs) was unaffected, which is why it only broke in CI. All subprocesses (`zenity` in `gui.py`, ffmpeg in `audio.py`) are spawned with `eo_stt.subprocess_env()` (`eo_stt/__init__.py`), which strips `LD_*`/`DYLD_*` loader vars; zenity failures (exit code not 0/1, or OSError) log a warning and fall back to the Tk dialog, and a failing Tk dialog shows an error message instead of failing silently. Regression test: `tests/test_gui.py::test_zenity_child_gets_clean_env` (real subprocess, fake zenity, asserts the child env is clean).

## Verified end-to-end (Oct 2026, CPU)

- INT8 ONNX model (local files, `quantization="int8"`) transcribes real Esperanto speech (first 40 s of `fremdulo.mp3`) correctly with token-derived word timestamps; quality is on par with (slightly better than) the old NeMo CTC model on the same audio.
- **Full 48-min podcast** (`fremdulo.mp3`, 2894.6 s) → `eo-stt-cli` → srt covering the full file (see e2e run), ~3-4 min on CPU.
- Accepted inputs tested: mp3 (CBR/VBR, 44.1 kHz mono), wav, ogg.
- Tests: 61 passed (audio incl. synthetic mp3 + anti-alias + blocked-resample, srt, transcribe pipeline with StubModel, GUI helpers incl. frozen-app subprocess-env regression tests, file-type dropdown and progress/status tests, UI-string ASCII guard, ONNX glue + real-model smoke test).
- Frozen-build dialogs: reproduced the v0.1.0 CI Linux artifact's broken file dialogs via a fake-ffmpeg env dump from the frozen app (child got `LD_LIBRARY_PATH=<bundle>/_internal`; `LD_LIBRARY_PATH=<ci _internal> zenity` crashes with a symbol lookup error, clean env opens). Fixed with `eo_stt.subprocess_env()`; verified on a local rebuild (child env clean) and by `tests/test_gui.py::test_zenity_child_gets_clean_env`.

## Portable builds (PyInstaller)

- `build.py` generates `eo-stt.spec` (gitignored byproduct), runs PyInstaller **onedir** for **this OS only** (PyInstaller cannot cross-compile): `.venv/bin/python build.py` → `dist/eo-stt/` (~250-350 MB).
- The exe is **named with the version**: `dist/eo-stt/eo-stt-v0.2-linux` (Linux) / `dist/eo-stt/eo-stt-v0.2.exe` (Windows). **no args → GUI; with args → CLI** (`eo-stt-v0.2-linux FILE.mp3 -o OUT_DIR`). Entry point is root `main.py` (kept outside the package for PyInstaller). Renaming the onedir exe is safe: it locates `_internal/` and `model/` relative to its own path, not its name. The Linux name ends in `-linux`, **not** in a digit: a trailing `.<digit>` (e.g. `eo-stt-v0.2`) matches the `*.1`–`*.9` man-page globs of the shared MIME database (`application/x-troff-man`, a text type), so Linux file managers open the binary in a text editor on double-click instead of running it (the exec bit is unaffected — verified with `xdg-mime query filetype`; regression test: `tests/test_build.py`).
- `.github/workflows/build.yml` runs pytest, then builds Linux (`tar.gz`) and Windows (`zip`) artifacts on `v*` tags (or manual `workflow_dispatch`), fetches the model first (cached), and attaches them to a GitHub release as `eo-stt-<tag>-linux.tar.gz` / `eo-stt-<tag>-windows.zip`. CI passes the tag to `build.py --version`, so the exe inside always matches the release. The archives contain the bundle's **contents directly** (versioned exe, `_internal/`, `model/` at the archive root) — no top-level `eo-stt/` folder. CI uses python.org Python (bundles Tk 8.6 — see font note). macOS is not built (app targets Linux/Windows); re-add `macos-latest` to the matrix if ever needed.
- The INT8 model (~142 MB) **is bundled** at `dist/eo-stt/model/` (beside the exe, copied by build.py after the PyInstaller run — not duplicated inside `_internal/`). Fully offline at runtime.
- PyInstaller flags that matter (in `build.py`): `--collect-all` for customtkinter/onnx_asr/onnxruntime/soundfile/darkdetect (onnx_asr reads `preprocessors/data/*.onnx` + `fbanks.npz` via importlib.resources; onnxruntime's native libs; soundfile's libsndfile), `--copy-metadata` for onnx-asr/onnxruntime/customtkinter (onnx_asr reads its version via importlib.metadata), excludes for torch/nemo/scipy/pandas/matplotlib/IPython/jupyter/huggingface_hub/cuda/tensorrt/tkinter.test (belt-and-braces; the build venv doesn't have them).
- **Frozen-app startup gotcha (carried over)**: do NOT `--exclude-module unittest`/`pydoc` — `numpy.testing` imports unittest and the frozen app dies without it.
- **Windows gotcha (v0.1.1 bug, fixed)**: the bundle must NOT contain `ucrtbase.dll`. The PyInstaller bootloader pre-loads a bundled `_internal/ucrtbase.dll` before loading `python312.dll`, and on Windows 10/11 that non-system UCRT copy makes the Python DLL load fail — the exe dies with `Failed to load Python DLL '...\_internal\python312.dll'. LoadLibrary: Invalid access to memory location` (error 998) before any Python code runs. The UCRT is part of the OS since Windows 10; the system copy must be used. `build.py` filters `ucrtbase.dll` out of `a.binaries` in the generated spec. CI smoke-runs the built exe (`--help`) on both OSes as a regression guard. (The v0.1.1 Windows artifact was built by CI without ever running the exe — the bug shipped because the workflow only built and zipped.)
- GUI is **CustomTkinter** (dark mode). Caveats on this machine's Linux/Xwayland + Tk: (1) Tk named fonts only resolve core X11 fonts, so CTk's TrueType fonts render as the bitmap "fixed" font (pixelated text; diacritics still show) — cosmetic only; (2) **set `corner_radius=0` on all CTk widgets** — CTk's rounded corners render as jagged notches here (broken-looking buttons); sharp rectangles render cleanly. On Windows/macOS (native font/canvas backends) CTk renders properly.

## Workspace context

- `../vocx` (Go) is the inverse tool: Esperanto text → phonetic Polish for TTS engines. Same domain (Esperanto audio tooling), independent stack — not a dependency.
