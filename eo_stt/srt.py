"""Subtitle (.srt) generation from transcribed text.

The ASR model outputs plain lowercase text (no punctuation), so subtitle
blocks are built from word counts and timing rules rather than sentence
boundaries. Two strategies:

- segments_from_words(): groups word-level timestamps into srt blocks.
- proportional_segments(): fallback when no word timestamps are available —
  splits the text into word-count chunks and spreads them evenly over the audio.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from pathlib import Path

MAX_LINE_CHARS = 42      # standard srt line length
MAX_BLOCK_CHARS = 84     # two lines per block
MAX_BLOCK_SECONDS = 7.0  # typical max subtitle duration
MAX_GAP_SECONDS = 1.5    # silence gap that forces a new block
MIN_BLOCK_SECONDS = 0.5  # never emit a shorter block (borrow time from the next one)


@dataclass
class Segment:
    start: float
    end: float
    text: str


def format_timestamp(seconds: float) -> str:
    """Format seconds as an srt timestamp: HH:MM:SS,mmm."""
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def segments_from_words(
    words: list[tuple[str, float, float]],
    max_chars: int = MAX_BLOCK_CHARS,
    max_seconds: float = MAX_BLOCK_SECONDS,
    max_gap: float = MAX_GAP_SECONDS,
) -> list[Segment]:
    """Group (word, start, end) tuples into srt blocks.

    A new block starts when the block would exceed max_chars, when it has been
    running longer than max_seconds, or when there is a silence gap > max_gap.
    """
    blocks: list[Segment] = []
    cur_words: list[str] = []
    cur_start = 0.0
    cur_end = 0.0
    cur_chars = 0
    prev_end: float | None = None
    for word, start, end in words:
        gap = (start - prev_end) if prev_end is not None else 0.0
        break_block = bool(cur_words) and (
            cur_chars + 1 + len(word) > max_chars
            or start - cur_start >= max_seconds
            or gap > max_gap
        )
        if break_block:
            blocks.append(Segment(cur_start, max(cur_end, cur_start + MIN_BLOCK_SECONDS), " ".join(cur_words)))
            cur_words, cur_chars = [], 0
        if not cur_words:
            cur_start = start
        cur_words.append(word)
        cur_chars += len(word) if cur_chars == 0 else 1 + len(word)
        cur_end = end
        prev_end = end
    if cur_words:
        blocks.append(Segment(cur_start, max(cur_end, cur_start + MIN_BLOCK_SECONDS), " ".join(cur_words)))
    return blocks


def proportional_segments(
    text: str,
    start: float = 0.0,
    duration: float | None = None,
    max_chars: int = MAX_BLOCK_CHARS,
) -> list[Segment]:
    """Split text into word-count chunks and spread them evenly over [start, start+duration]."""
    words = text.split()
    if not words:
        return []
    if duration is None:
        duration = max(len(words) * 0.4, MIN_BLOCK_SECONDS)  # rough speech-rate fallback
    blocks: list[list[str]] = []
    cur: list[str] = []
    cur_len = 0
    for w in words:
        add = len(w) if not cur else 1 + len(w)
        if cur and cur_len + add > max_chars:
            blocks.append(cur)
            cur, cur_len = [], 0
            add = len(w)
        cur.append(w)
        cur_len += add
    if cur:
        blocks.append(cur)
    weights = [sum(len(w) + 1 for w in b) for b in blocks]
    total_weight = sum(weights)
    segments: list[Segment] = []
    t = start
    for block, weight in zip(blocks, weights):
        dur = duration * weight / total_weight
        segments.append(Segment(t, t + dur, " ".join(block)))
        t += dur
    segments[-1].end = start + duration  # absorb rounding drift
    _enforce_min_duration(segments)
    return segments


def _enforce_min_duration(segments: list[Segment], minimum: float = MIN_BLOCK_SECONDS) -> None:
    """Lengthen too-short blocks by borrowing time from the following block."""
    for i, seg in enumerate(segments[:-1]):
        if seg.end - seg.start < minimum:
            need = minimum - (seg.end - seg.start)
            nxt = segments[i + 1]
            if nxt.end - nxt.start - need >= minimum:
                seg.end += need
                nxt.start += need


def format_block(index: int, seg: Segment) -> list[str]:
    """One srt block as lines: index, timestamp line, wrapped text."""
    lines = [str(index), f"{format_timestamp(seg.start)} --> {format_timestamp(seg.end)}"]
    lines.extend(textwrap.wrap(seg.text, MAX_LINE_CHARS) or [""])
    return lines


def write_srt(segments: list[Segment], path: str | Path) -> Path:
    """Write segments to an .srt file (UTF-8). Returns the path."""
    if not segments:
        raise ValueError("no segments to write")
    path = Path(path)
    lines: list[str] = []
    for i, seg in enumerate(segments, 1):
        lines.extend(format_block(i, seg))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


class SrtWriter:
    """Incrementally write an .srt file as segments arrive.

    The file is created (truncated) immediately; blocks are appended and
    flushed as they are produced, so the .srt can be opened and read while
    the transcription that produces it is still running.
    """

    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._fh = self._path.open("w", encoding="utf-8")
        self._index = 0

    def append(self, segments: list[Segment]) -> None:
        """Append segments as new blocks (numbered sequentially)."""
        for seg in segments:
            self._index += 1
            self._fh.write("\n".join(format_block(self._index, seg)))
            self._fh.write("\n\n")
        if segments:
            self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "SrtWriter":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
