import numpy as np
import pytest
import soundfile as sf

from eo_stt.audio import TARGET_SAMPLE_RATE, audio_info, load_audio


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
