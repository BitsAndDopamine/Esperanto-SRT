"""End-to-end pipeline: audio file -> .srt file."""

from __future__ import annotations

from pathlib import Path

from .audio import load_audio
from .model import transcribe_chunks
from .srt import Segment, proportional_segments, segments_from_words, write_srt


def transcribe_to_segments(model, audio_path: str | Path) -> list[Segment]:
    """Transcribe an audio file and return timed subtitle segments."""
    samples = load_audio(audio_path)
    segments: list[Segment] = []
    for text, words, start, duration in transcribe_chunks(model, samples):
        if words:
            # real word timestamps from the model, offset into the full file
            for block in segments_from_words(words):
                segments.append(Segment(start + block.start, start + block.end, block.text))
        else:
            # fallback: spread the text evenly over the chunk
            segments.extend(proportional_segments(text, start, duration))
    return segments


def srt_path_for(audio_path: str | Path, output_dir: str | Path | None = None) -> Path:
    """Default .srt output path: same stem, either in output_dir or next to the input."""
    audio_path = Path(audio_path)
    out_dir = Path(output_dir) if output_dir else audio_path.parent
    return out_dir / (audio_path.stem + ".srt")


def transcribe_file(model, audio_path: str | Path, output_dir: str | Path | None = None) -> Path:
    """Transcribe one audio file to .srt. Returns the written path."""
    segments = transcribe_to_segments(model, audio_path)
    out_path = srt_path_for(audio_path, output_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return write_srt(segments, out_path)
