"""End-to-end pipeline: audio file -> .srt file."""

from __future__ import annotations

import logging
from pathlib import Path

from .audio import TARGET_SAMPLE_RATE, load_audio
from .model import transcribe_chunks
from .srt import Segment, SrtWriter, proportional_segments, segments_from_words

logger = logging.getLogger("eo_stt")


def transcribe_to_segments(model, audio_path: str | Path, on_progress=None, on_segments=None) -> list[Segment]:
    """Transcribe an audio file and return timed subtitle segments.

    on_progress(done, total) is called with chunk progress (see model.transcribe_chunks).
    on_segments(new_segments) is called with the segments of each chunk as soon
    as that chunk is transcribed, so callers can write them out incrementally.
    """
    samples = load_audio(audio_path)
    duration = len(samples) / TARGET_SAMPLE_RATE
    segments: list[Segment] = []
    for text, words, start, chunk_duration in transcribe_chunks(model, samples, on_progress=on_progress):
        chunk_segments: list[Segment] = []
        if words:
            # real word timestamps from the model, offset into the full file
            for block in segments_from_words(words):
                chunk_segments.append(Segment(start + block.start, start + block.end, block.text))
        else:
            # fallback: spread the text evenly over the chunk
            chunk_segments.extend(proportional_segments(text, start, chunk_duration))
        segments.extend(chunk_segments)
        if on_segments is not None:
            on_segments(chunk_segments)
    if duration > 60.0 and (not segments or segments[-1].end < 0.5 * duration):
        covered = segments[-1].end if segments else 0.0
        logger.warning(
            "transskribo kovras nur %.0f%% de %.0f-sekunda dosiero (%.0fs); "
            "kelkaj pecoj verŝajne malsukcesis - kontrolu la protokolon pri eraroj.",
            100.0 * covered / duration,
            duration,
            covered,
        )
    return segments


def srt_path_for(audio_path: str | Path, output_dir: str | Path | None = None) -> Path:
    """Default .srt output path: same stem, either in output_dir or next to the input."""
    audio_path = Path(audio_path)
    out_dir = Path(output_dir) if output_dir else audio_path.parent
    return out_dir / (audio_path.stem + ".srt")


def transcribe_file(model, audio_path: str | Path, output_dir: str | Path | None = None, on_progress=None) -> Path:
    """Transcribe one audio file to .srt. Returns the written path.

    The .srt is written incrementally: the file is created up front and grows
    block by block while transcription runs, so it can be opened and checked
    meanwhile. If transcription fails midway, the partial file is kept.
    """
    out_path = srt_path_for(audio_path, output_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with SrtWriter(out_path) as writer:
        segments = transcribe_to_segments(model, audio_path, on_progress=on_progress, on_segments=writer.append)
    if not segments:
        out_path.unlink(missing_ok=True)  # don't leave an empty file behind
        raise ValueError("no segments to write")
    return out_path
