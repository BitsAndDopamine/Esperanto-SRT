"""Audio loading and preprocessing.

The ASR model expects 16 kHz mono audio. All decoding/resampling happens here,
so the rest of the app only ever sees float32 numpy arrays at 16 kHz.

Decoding relies on soundfile (libsndfile >= 1.1 reads mp3, wav, ogg and flac
directly — no ffmpeg needed). If soundfile fails on a file, we fall back to an
ffmpeg subprocess when ffmpeg happens to be installed.

Resampling is a small numpy-only polyphase resampler (anti-aliased Kaiser-
windowed sinc FIR, same design as scipy.signal.resample_poly) so the app does
not need scipy — a ~100 MB installed dependency — just for this.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf

from . import subprocess_env

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
        mono = _resample_poly(mono, TARGET_SAMPLE_RATE // g, int(sr) // g)
    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    if peak > 1.0:
        mono = mono / peak
    return np.ascontiguousarray(mono, dtype=np.float32)


def _resample_poly(x: np.ndarray, up: int, down: int) -> np.ndarray:
    """Anti-aliased polyphase resampling (numpy-only replacement for scipy.signal.resample_poly).

    Zero-stuffs by `up`, applies a Kaiser-windowed sinc low-pass (cutoff at the
    lower Nyquist frequency, 10*max(up, down) half-length like scipy's default),
    then decimates by `down`. `up` and `down` must be coprime. Runs in blocks,
    so arbitrarily long recordings use bounded memory.
    """
    if up == down:
        return x.astype(np.float32)
    max_rate = max(up, down)
    half_len = 10 * max_rate
    n_taps = 2 * half_len + 1
    m = np.arange(-half_len, half_len + 1, dtype=np.float64)
    cutoff = 1.0 / max_rate  # firwin-style: 1.0 = Nyquist of the upsampled rate
    h = cutoff * np.sinc(cutoff * m) * np.kaiser(n_taps, 5.0)
    h /= h.sum()  # unit gain at DC
    h *= up  # compensate zero-stuffing
    h = h.astype(np.float32)

    # polyphase decomposition: phase p (0..up-1) uses taps h[p::up]
    # (zero-pad the filter so every phase has exactly taps_per_phase taps)
    taps_per_phase = -(-n_taps // up)  # ceil(n_taps / up)
    h_pad = np.zeros(taps_per_phase * up, dtype=np.float32)
    h_pad[:n_taps] = h
    phases = np.zeros((up, taps_per_phase), dtype=np.float32)
    for p in range(up):
        phases[p] = h_pad[p::up]

    n_in = x.shape[0]
    n_out = int(np.ceil(n_in * up / down))
    x32 = np.ascontiguousarray(x, dtype=np.float32)

    # output n: y[n] = sum_j phases[p_n, j] * x[base_n - j]
    # with p_n = (n*down) % up and base_n = (n*down) // up
    ns = np.arange(n_out, dtype=np.int64)
    bases = (ns * down) // up  # int64: input indices can exceed int32 for very long files
    ps = ((ns * down) % up).astype(np.int32)

    # zero-pad so the first/last outputs see the full filter transient
    pad = taps_per_phase - 1
    xp = np.concatenate([np.zeros(pad, np.float32), x32, np.zeros(pad, np.float32)])
    jj = np.arange(taps_per_phase - 1, -1, -1, dtype=np.int32)  # reversed tap index

    y = np.empty(n_out, dtype=np.float32)
    block = 1 << 16  # outputs per block (bounds the gather matrix to ~tens of MB)
    for start in range(0, n_out, block):
        stop = min(start + block, n_out)
        idx = bases[start:stop, None] + np.int32(pad) - jj[None, :]
        w = xp[idx]  # (block, taps_per_phase)
        y[start:stop] = np.einsum("ij,ij->i", w, phases[ps[start:stop]][:, jj])
    return y


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
            env=subprocess_env(),  # no LD_LIBRARY_PATH from a frozen build (see eo_srt.subprocess_env)
        )
        data, sr = sf.read(out, dtype="float32", always_2d=True)
        return data, sr
    finally:
        Path(out).unlink(missing_ok=True)
