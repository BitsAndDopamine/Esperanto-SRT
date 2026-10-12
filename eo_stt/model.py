"""NeMo ASR model: loading and transcription.

Heavy imports (torch/NeMo) are lazy so that the GUI starts fast and the pure
modules (audio/srt) stay testable without NeMo installed.

Word-level timestamps: NeMo's CTC transcribe() supports timestamps=True, which
returns Hypothesis objects whose .timestamp['word'] is a list of
{"word", "start_offset", "end_offset", "start", "end"} dicts (start/end in
seconds, converted from frames by NeMo itself).

Audio is passed to NeMo as numpy arrays (in-process tensor dataloader,
num_workers=0) — NOT as file paths. The file-path route goes through a Lhotse
manifest dataloader that spawns worker processes per call, which is fragile
when transcription runs in the GUI's worker thread (and very slow on Windows,
where multiprocessing uses spawn). The tensor route has no worker processes.
"""

from __future__ import annotations

import logging

import numpy as np

from .audio import TARGET_SAMPLE_RATE

logger = logging.getLogger("eo_stt")

MODEL_ID = "nvidia/stt_eo_conformer_ctc_large"
CHUNK_SECONDS = 30.0  # long audio is transcribed in chunks (bounded memory, CPU-friendly)
BATCH_CHUNKS = 4      # chunks per model.transcribe call (batched inference)

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
    batch_chunks: int = BATCH_CHUNKS,
    on_progress=None,
) -> list[ChunkResult]:
    """Transcribe a 16 kHz mono array in chunks of at most chunk_seconds.

    Chunks are sent to the model in batches of batch_chunks (one transcribe
    call per batch). on_progress(done, total) is called after each chunk.
    """
    n_chunk = int(sample_rate * chunk_seconds)
    total = len(samples)
    if total == 0:
        return [("", [], 0.0, 0.0)]
    chunks = [samples[i : i + n_chunk] for i in range(0, total, n_chunk)]
    texts_words: list[tuple[str, list[tuple[str, float, float]]]] = []
    done = 0
    for b in range(0, len(chunks), batch_chunks):
        batch = chunks[b : b + batch_chunks]
        try:
            hyps = _transcribe_arrays(model, batch)
        except Exception as exc:
            # isolate: retry one chunk per call so a single bad chunk can't kill the file
            logger.warning("batch transcription failed (%s); retrying %d chunk(s) individually", exc, len(batch))
            hyps = []
            for single in batch:
                hyps.extend(_transcribe_arrays(model, [single]))
        for hyp in hyps:
            text = hyp.text if hasattr(hyp, "text") else str(hyp)
            texts_words.append((text.strip(), _word_timestamps(hyp)))
            done += 1
            if on_progress is not None:
                on_progress(done, len(chunks))
    results: list[ChunkResult] = []
    for idx, (text, words) in enumerate(texts_words):
        start = idx * chunk_seconds
        duration = len(chunks[idx]) / sample_rate
        results.append((text, words, start, duration))
    return results


def _transcribe_arrays(model, arrays: list[np.ndarray]):
    """One batched transcribe call. Arrays must be 16 kHz mono float32."""
    try:
        return model.transcribe(list(arrays), batch_size=len(arrays), timestamps=True, verbose=False, num_workers=0)
    except TypeError:  # older NeMo without the timestamps/num_workers kwargs
        return model.transcribe(list(arrays), batch_size=len(arrays), verbose=False)


def _word_timestamps(hyp) -> list[tuple[str, float, float]]:
    """Extract [(word, start, end), ...] in seconds from a NeMo Hypothesis."""
    timestamp = getattr(hyp, "timestamp", None) or {}
    words = []
    for entry in timestamp.get("word") or []:
        word = str(entry.get("word", "")).strip()
        if word:
            words.append((word, float(entry["start"]), float(entry["end"])))
    return words
