"""ONNX ASR model: loading and transcription (CPU-only, fully offline).

Backend: onnx-asr + onnxruntime running the INT8 ONNX export of NVIDIA's
Esperanto Conformer-Transducer Large model
(OpenVoiceOS/nvidia-eo-conformer-transducer-large-onnx, license CC-BY-4.0).

The model is loaded exclusively from local files (see default_model_dir) —
the app never downloads anything at runtime. Heavy imports (onnxruntime /
onnx_asr) are lazy so that the GUI starts fast and the pure modules
(audio/srt) stay testable without the inference stack installed.

Word-level timestamps: the transducer emits one timestamp per token (seconds,
0.01 s hop x 4 subsampling = 0.04 s per encoder frame). _words_from_tokens
regroups the SentencePiece tokens into words and turns token spans into
(word, start, end) tuples in seconds, exposed through hypothesis objects that
mimic the NeMo fields the rest of the app uses (.text, .timestamp['word']).

Audio is passed to the recognizer as numpy arrays (16 kHz mono float32) —
never as file paths — so no worker processes or temp files are involved.
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from .audio import TARGET_SAMPLE_RATE

logger = logging.getLogger("eo_stt")

MODEL_ID = "OpenVoiceOS/nvidia-eo-conformer-transducer-large-onnx"
MODEL_DIR_NAME = "nvidia-eo-conformer-transducer-large-onnx-int8"
QUANTIZATION = "int8"
CHUNK_SECONDS = 30.0  # long audio is transcribed in chunks (bounded memory, CPU-friendly)
BATCH_CHUNKS = 4      # chunks per recognizer call (batched encoder inference)

# one encoder frame = 0.01 s hop x 4 subsampling (see config.json); token
# timestamps from onnx-asr are frame_index * this value, in seconds
FRAME_SECONDS = 0.04

# (text, words, start_seconds, duration_seconds); words = [(word, start, end), ...]
# in seconds relative to the chunk start, [] if timestamps are unavailable
ChunkResult = tuple[str, list[tuple[str, float, float]], float, float]

_model_cache: dict[str, object] = {}


def default_model_dir() -> Path:
    """Locate the local INT8 model directory (never downloads).

    Order: $EO_STT_MODEL_DIR, then (frozen app) <exe dir>/model or the
    PyInstaller _MEIPASS bundle, then (source tree) <repo>/models/<name>.
    """
    env = os.environ.get("EO_STT_MODEL_DIR")
    if env:
        return Path(env)
    candidates: list[Path] = []
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        candidates.append(exe_dir / "model")
        meipass = Path(getattr(sys, "_MEIPASS", exe_dir))
        candidates.append(meipass / "model")
    else:
        candidates.append(Path(__file__).resolve().parent.parent / "models" / MODEL_DIR_NAME)
    for cand in candidates:
        if (cand / "config.json").is_file():
            return cand
    return candidates[0]


def load_model(model_dir: str | Path | None = None):
    """Load the ASR model from local files (cached per process). CPU-only.

    Raises RuntimeError with an actionable message if the model files are
    missing or invalid.
    """
    model_dir = Path(model_dir) if model_dir is not None else default_model_dir()
    key = str(model_dir)
    if key not in _model_cache:
        logger.info("preparante modelon (%s ONNX) de %s", QUANTIZATION, model_dir)
        _model_cache[key] = OnnxAsrModel(model_dir)
    return _model_cache[key]


class OnnxAsrModel:
    """INT8 ONNX recognizer with the transcribe() surface the app used with NeMo.

    transcribe(arrays, timestamps=True) takes a list of 16 kHz mono float32
    arrays and returns one hypothesis-like object per array.
    """

    def __init__(self, model_dir: Path):
        import onnx_asr  # lazy: heavy import, only needed when transcribing

        try:
            adapter = onnx_asr.load_model(
                MODEL_ID,
                path=str(model_dir),
                quantization=QUANTIZATION,
                providers=["CPUExecutionProvider"],
            )
        except Exception as exc:
            raise RuntimeError(
                f"Cannot load the ASR model from '{model_dir}': {exc} "
                "The app needs the local INT8 model files (encoder-model.int8.onnx, "
                "decoder_joint-model.int8.onnx, config.json, vocab.txt). "
                "Run 'python fetch_model.py' to download them, or point "
                "EO_STT_MODEL_DIR at a directory that contains them."
            ) from exc
        self._recognizer = adapter.with_timestamps()
        self.model_dir = model_dir

    def transcribe(self, arrays: list[np.ndarray], timestamps: bool = True, **_kwargs):
        arrays = [np.asarray(a, dtype=np.float32) for a in arrays]
        results = self._recognizer.recognize(arrays, sample_rate=TARGET_SAMPLE_RATE)
        hyps = []
        for res in results:
            words = (
                _words_from_tokens(res.tokens or [], res.timestamps or [])
                if timestamps
                else []
            )
            hyps.append(_Hypothesis(res.text, words))
        return hyps


class _Hypothesis:
    """Mimics the NeMo Hypothesis fields the app uses: .text and .timestamp['word']."""

    def __init__(self, text: str, words: list[tuple[str, float, float]]):
        self.text = text
        self.timestamp = {"word": [{"word": w, "start": s, "end": e} for w, s, e in words]}


def _words_from_tokens(
    tokens: list[str],
    timestamps: list[float],
    frame_seconds: float = FRAME_SECONDS,
) -> list[tuple[str, float, float]]:
    """Regroup SentencePiece tokens into (word, start, end) tuples in seconds.

    onnx-asr decodes ▁ as a leading space, so a token starting with a space
    begins a new word. A token emitted at frame t covers [t, next token's t);
    the last token covers one extra frame. Words are relative to the chunk.
    """
    words: list[tuple[str, float, float]] = []
    cur_text: list[str] = []
    cur_start = 0.0
    cur_end = 0.0
    n = len(tokens)
    for i, tok in enumerate(tokens):
        start = float(timestamps[i])
        end = float(timestamps[i + 1]) if i + 1 < n else start + frame_seconds
        if tok.startswith(" ") or not cur_text:
            if cur_text:
                words.append(("".join(cur_text), cur_start, cur_end))
            cur_text = [tok[1:] if tok.startswith(" ") else tok]
            cur_start, cur_end = start, end
        else:
            cur_text.append(tok)
            cur_end = end
    if cur_text:
        words.append(("".join(cur_text), cur_start, cur_end))
    return [(w.strip(), s, e) for w, s, e in words if w.strip()]


def transcribe_chunks(
    model,
    samples: np.ndarray,
    sample_rate: int = TARGET_SAMPLE_RATE,
    chunk_seconds: float = CHUNK_SECONDS,
    batch_chunks: int = BATCH_CHUNKS,
    on_progress=None,
) -> Iterator[ChunkResult]:
    """Transcribe a 16 kHz mono array in chunks of at most chunk_seconds.

    Yields (text, words, start, duration) per chunk as soon as its batch is
    transcribed, so callers can consume (and write out) chunks incrementally.
    Chunks are sent to the model in batches of batch_chunks (one transcribe
    call per batch). on_progress(done, total) is called after each chunk.
    """
    n_chunk = int(sample_rate * chunk_seconds)
    total = len(samples)
    if total == 0:
        yield ("", [], 0.0, 0.0)
        return
    chunks = [samples[i : i + n_chunk] for i in range(0, total, n_chunk)]
    done = 0
    for b in range(0, len(chunks), batch_chunks):
        batch = chunks[b : b + batch_chunks]
        try:
            hyps = _transcribe_arrays(model, batch)
        except Exception as exc:
            # isolate: retry one chunk per call so a single bad chunk can't kill the file
            logger.warning("grupa transskribo malsukcesis (%s); reprovante %d pecojn unuope", exc, len(batch))
            hyps = []
            for single in batch:
                hyps.extend(_transcribe_arrays(model, [single]))
        for chunk, hyp in zip(batch, hyps):
            text = hyp.text if hasattr(hyp, "text") else str(hyp)
            words = _word_timestamps(hyp)
            start = done * chunk_seconds
            duration = len(chunk) / sample_rate
            done += 1
            if on_progress is not None:
                on_progress(done, len(chunks))
            yield (text.strip(), words, start, duration)


def _transcribe_arrays(model, arrays: list[np.ndarray]):
    """One batched transcribe call. Arrays must be 16 kHz mono float32."""
    return model.transcribe(list(arrays), timestamps=True)


def _word_timestamps(hyp) -> list[tuple[str, float, float]]:
    """Extract [(word, start, end), ...] in seconds from a hypothesis."""
    timestamp = getattr(hyp, "timestamp", None) or {}
    words = []
    for entry in timestamp.get("word") or []:
        word = str(entry.get("word", "")).strip()
        if word:
            words.append((word, float(entry["start"]), float(entry["end"])))
    return words
