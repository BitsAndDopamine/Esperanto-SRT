"""NeMo ASR model: loading and transcription.

Heavy imports (torch/NeMo) are lazy so that the GUI starts fast and the pure
modules (audio/srt) stay testable without NeMo installed.

Word-level timestamps: NeMo's CTC transcribe() supports timestamps=True, which
returns Hypothesis objects whose .timestamp['word'] is a list of
{"word", "start_offset", "end_offset", "start", "end"} dicts (start/end in
seconds, converted from frames by NeMo itself).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from .audio import TARGET_SAMPLE_RATE

MODEL_ID = "nvidia/stt_eo_conformer_ctc_large"
CHUNK_SECONDS = 30.0  # long audio is transcribed in chunks (bounded memory, CPU-friendly)

# (text, words, start_seconds, duration_seconds); words = [(word, start, end), ...]
# in seconds relative to the chunk start, [] if timestamps are unavailable
ChunkResult = tuple[str, list[tuple[str, float, float]], float, float]

_model_cache: dict[tuple[str, str], object] = {}


def load_model(model_id: str = MODEL_ID, device: str = "auto"):
    """Load the ASR model (cached per process).

    The first call downloads the checkpoint (~500 MB) into the Hugging Face
    cache (~/.cache/huggingface on Linux). device: "cpu", "cuda" or "auto".
    """
    import torch

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    key = (model_id, device)
    if key not in _model_cache:
        from omegaconf import open_dict

        import nemo.collections.asr as nemo_asr

        model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_id)
        model.eval()
        # greedy_batch is NeMo's recommended CTC strategy: same output as greedy, faster
        with open_dict(model.cfg.decoding):
            model.cfg.decoding.strategy = "greedy_batch"
        if device != "cpu":
            model = model.to(device)
        _model_cache[key] = model
    return _model_cache[key]


def transcribe_chunks(
    model,
    samples: np.ndarray,
    sample_rate: int = TARGET_SAMPLE_RATE,
    chunk_seconds: float = CHUNK_SECONDS,
) -> list[ChunkResult]:
    """Transcribe a 16 kHz mono array in chunks of at most chunk_seconds."""
    n_chunk = int(sample_rate * chunk_seconds)
    total = len(samples)
    if total == 0:
        return [("", [], 0.0, 0.0)]
    chunks = []
    for i in range(0, total, n_chunk):
        part = samples[i : i + n_chunk]
        text, words = _transcribe_array(model, part)
        chunks.append((text, words, i / sample_rate, len(part) / sample_rate))
    return chunks


def _transcribe_array(model, samples: np.ndarray) -> tuple[str, list[tuple[str, float, float]]]:
    """Transcribe one chunk. NeMo's transcribe() takes file paths, so write a temp wav."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        sf.write(tmp_path, samples, TARGET_SAMPLE_RATE)
        try:
            result = model.transcribe([tmp_path], batch_size=1, timestamps=True, verbose=False)
        except TypeError:  # older NeMo without timestamp support
            result = model.transcribe([tmp_path], batch_size=1)
        hyp = result[0]
        text = hyp.text if hasattr(hyp, "text") else str(hyp)
        words = _word_timestamps(hyp)
        return text.strip(), words
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _word_timestamps(hyp) -> list[tuple[str, float, float]]:
    """Extract [(word, start, end), ...] in seconds from a NeMo Hypothesis."""
    timestamp = getattr(hyp, "timestamp", None) or {}
    words = []
    for entry in timestamp.get("word") or []:
        word = str(entry.get("word", "")).strip()
        if word:
            words.append((word, float(entry["start"]), float(entry["end"])))
    return words
