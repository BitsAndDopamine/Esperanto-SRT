import subprocess

import numpy as np
import pytest
import soundfile as sf

import eo_srt.audio as audio
from eo_srt.audio import TARGET_SAMPLE_RATE, audio_info, load_audio


def _write_wav(path, sr=44100, seconds=2.0, freq=440.0, channels=2):
    t = np.arange(int(sr * seconds)) / sr
    sig = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    data = np.stack([sig] * channels, axis=1) if channels > 1 else sig
    sf.write(str(path), data, sr)
    return path


def test_load_audio_resamples_to_16k_mono(tmp_path):
    p = _write_wav(tmp_path / "a.wav", sr=44100, channels=2)
    x = load_audio(p)
    assert x.ndim == 1
    assert abs(len(x) / TARGET_SAMPLE_RATE - 2.0) < 0.05
    assert x.dtype == np.float32


def test_load_audio_preserves_pitch(tmp_path):
    p = _write_wav(tmp_path / "a.wav", sr=44100, freq=440.0)
    x = load_audio(p)
    spec = np.abs(np.fft.rfft(x[:16000]))
    freqs = np.fft.rfftfreq(16000, 1 / 16000)
    assert abs(freqs[int(np.argmax(spec))] - 440.0) < 20


def test_audio_info(tmp_path):
    p = _write_wav(tmp_path / "a.wav", sr=22050, seconds=1.5, channels=1)
    duration, sr, channels = audio_info(p)
    assert duration == pytest.approx(1.5, abs=0.01)
    assert sr == 22050
    assert channels == 1


def test_load_audio_mp3(tmp_path):
    lameenc = pytest.importorskip("lameenc")
    wav = _write_wav(tmp_path / "a.wav", sr=44100, seconds=2.0, channels=1)
    data, sr = sf.read(str(wav), dtype="int16")
    enc = lameenc.Encoder()
    enc.set_bit_rate(128)
    enc.set_in_sample_rate(sr)
    enc.set_channels(1)
    enc.set_quality(2)
    mp3_path = tmp_path / "a.mp3"
    mp3_path.write_bytes(enc.encode(data.tobytes()) + enc.flush())
    x = load_audio(mp3_path)
    assert x.ndim == 1
    # mp3 adds encoder delay/padding — allow generous tolerance
    assert abs(len(x) / TARGET_SAMPLE_RATE - 2.0) < 0.25


def test_load_audio_resamples_common_rates(tmp_path):
    # 48 kHz (down), 22.05 kHz (down), 8 kHz (up) — all must land on 16 kHz mono
    for sr in (48000, 22050, 8000):
        p = _write_wav(tmp_path / f"a{sr}.wav", sr=sr, seconds=1.5, channels=1)
        x = load_audio(p)
        assert x.ndim == 1 and x.dtype == np.float32
        assert abs(len(x) / TARGET_SAMPLE_RATE - 1.5) < 0.05, sr


def test_load_audio_resample_antialiases(tmp_path):
    # 6 kHz tone (passband) + 12 kHz tone (above the 16 kHz Nyquist, would
    # alias to 4 kHz without an anti-alias filter) — the alias must be weak
    sr = 44100
    t = np.arange(int(sr * 1.0)) / sr
    sig = (0.5 * np.sin(2 * np.pi * 6000.0 * t) + 0.5 * np.sin(2 * np.pi * 12000.0 * t)).astype(np.float32)
    p = tmp_path / "tone.wav"
    sf.write(str(p), sig, sr)
    x = load_audio(p)

    def power_at(freq):
        seg = x[1600:-1600]
        n = np.arange(len(seg))
        return abs(np.dot(seg, np.exp(-2j * np.pi * freq * n / TARGET_SAMPLE_RATE))) ** 2 / len(seg)

    assert power_at(6000.0) > 100 * power_at(4000.0)  # > 20 dB alias rejection


def test_load_audio_long_file_blocked_resample(tmp_path):
    # > 65536 output samples exercises the blocked polyphase path
    p = _write_wav(tmp_path / "long.wav", sr=44100, seconds=30.0, channels=1)
    x = load_audio(p)
    assert abs(len(x) / TARGET_SAMPLE_RATE - 30.0) < 0.1
    assert np.isfinite(x).all()


def test_ffmpeg_fallback_child_env_is_clean(tmp_path, monkeypatch):
    # a frozen build's bootloader points LD_LIBRARY_PATH at the bundled
    # _internal/ libraries; the ffmpeg child must not inherit that (see
    # eo_srt.subprocess_env)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/frozen/_internal")
    seen = {}

    def fake_run(cmd, **kwargs):
        seen.update(kwargs)
        sf.write(cmd[-1], np.zeros(TARGET_SAMPLE_RATE, dtype=np.float32), TARGET_SAMPLE_RATE)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(audio.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(audio.subprocess, "run", fake_run)
    data, sr = audio._read_with_ffmpeg(tmp_path / "x.m4a")
    assert sr == TARGET_SAMPLE_RATE
    assert len(data) == TARGET_SAMPLE_RATE
    env = seen.get("env", {})
    assert "LD_LIBRARY_PATH" not in env
    assert "PATH" in env  # the rest of the environment is preserved
