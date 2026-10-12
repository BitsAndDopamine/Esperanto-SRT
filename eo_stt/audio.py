"""Audio loading and preprocessing.

The NeMo model expects 16 kHz mono audio. All decoding/resampling happens here,
so the rest of the app only ever sees float32 numpy arrays at 16 kHz.

Decoding relies on soundfile (libsndfile >= 1.1 reads mp3, wav, ogg and flac
directly — no ffmpeg needed). If soundfile fails on a file, we fall back to an
ffmpeg subprocess when ffmpeg happens to be installed.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TARGET_SAMPLE_RATE = 16000


def audio_info(path: str | Path) -> tuple[float, int, int]:
    """Return (duration_seconds, sample_rate, channels) without decoding samples."""
    info = sf.info(str(path))
    return info.duration, info.samplerate, info.channels


def load_audio(path: str | Path) -> np.ndarray:
    """Load any supported audio file as mono float32 at 16 kHz."""
    path = Path(path)
    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception:
        data, sr = _read_with_ffmpeg(path)
    mono = data.mean(axis=1).astype(np.float32)
    if sr != TARGET_SAMPLE_RATE:
        g = gcd(int(sr), TARGET_SAMPLE_RATE)
        mono = resample_poly(mono, TARGET_SAMPLE_RATE // g, int(sr) // g).astype(np.float32)
    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    if peak > 1.0:
        mono = mono / peak
    return mono


def _read_with_ffmpeg(path: Path) -> tuple[np.ndarray, int]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            f"Cannot decode {path}: soundfile failed and ffmpeg is not installed. "
            "Install ffmpeg or convert the file to wav/mp3/ogg/flac first."
        )
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        out = tmp.name
    try:
        subprocess.run(
            [ffmpeg, "-y", "-i", str(path), "-ac", "1", "-ar", str(TARGET_SAMPLE_RATE), out],
            check=True,
            capture_output=True,
        )
        data, sr = sf.read(out, dtype="float32", always_2d=True)
        return data, sr
    finally:
        Path(out).unlink(missing_ok=True)
