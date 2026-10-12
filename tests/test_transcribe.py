import numpy as np
import pytest
import soundfile as sf

from eo_stt.transcribe import srt_path_for, transcribe_file, transcribe_to_segments


class StubModel:
    """Mimics the NeMo model's transcribe(): one result per input array."""

    def __init__(self, text="saluton mondo"):
        self.text = text
        self.calls = []

    def transcribe(self, arrays, **kwargs):
        self.calls.append(list(arrays))
        return [self.text for _ in arrays]


class BatchFlakyModel(StubModel):
    """Fails on multi-array batches, forcing the per-chunk fallback."""

    def transcribe(self, arrays, **kwargs):
        if len(arrays) > 1:
            raise RuntimeError("simulated batch failure")
        return super().transcribe(arrays, **kwargs)


def _write_silence(path, seconds, sr=16000):
    sf.write(str(path), np.zeros(int(sr * seconds), dtype=np.float32), sr)
    return path


def test_transcribe_to_segments(tmp_path):
    wav = _write_silence(tmp_path / "a.wav", 5.0)
    model = StubModel()
    segs = transcribe_to_segments(model, wav)
    assert len(model.calls) == 1  # one batch for one chunk
    assert segs
    assert segs[0].start == 0.0
    assert segs[-1].end == pytest.approx(5.0)
    assert segs[0].text == "saluton mondo"


def test_transcribe_chunks_long_file(tmp_path):
    wav = _write_silence(tmp_path / "long.wav", 65.0)  # -> 30s + 30s + 5s chunks
    model = StubModel()
    segs = transcribe_to_segments(model, wav)
    assert sum(len(c) for c in model.calls) == 3  # all 3 chunks were sent
    assert segs[0].start == 0.0
    assert segs[-1].end == pytest.approx(65.0)
    # chunk boundaries at 30s and 60s
    starts = sorted({round(s.start, 3) for s in segs})
    assert any(29.0 < s < 31.0 for s in starts)
    assert any(59.0 < s < 61.0 for s in starts)


def test_transcribe_chunks_batches(tmp_path):
    wav = _write_silence(tmp_path / "long.wav", 250.0)  # 9 chunks -> 3 batches of 4/4/1
    model = StubModel()
    transcribe_to_segments(model, wav)
    assert [len(c) for c in model.calls] == [4, 4, 1]


def test_transcribe_chunks_falls_back_to_single_chunks(tmp_path):
    wav = _write_silence(tmp_path / "long.wav", 65.0)  # 3 chunks
    model = BatchFlakyModel()
    segs = transcribe_to_segments(model, wav)
    assert len(model.calls) == 3  # batch failed -> 3 individual calls
    assert all(len(c) == 1 for c in model.calls)
    assert segs[-1].end == pytest.approx(65.0)


def test_on_progress(tmp_path):
    wav = _write_silence(tmp_path / "long.wav", 65.0)  # 3 chunks
    calls = []
    transcribe_to_segments(StubModel(), wav, on_progress=lambda d, t: calls.append((d, t)))
    assert calls == [(1, 3), (2, 3), (3, 3)]


def test_transcribe_file_writes_srt(tmp_path):
    wav = _write_silence(tmp_path / "a.wav", 3.0)
    out = transcribe_file(StubModel(), wav)
    assert out == tmp_path / "a.srt"
    content = out.read_text(encoding="utf-8")
    assert "00:00:00,000 -->" in content
    assert "saluton mondo" in content


def test_transcribe_file_output_dir(tmp_path):
    wav = _write_silence(tmp_path / "a.wav", 3.0)
    out_dir = tmp_path / "subs"
    out = transcribe_file(StubModel(), wav, out_dir)
    assert out == out_dir / "a.srt"
    assert out.is_file()


def test_srt_path_for():
    assert srt_path_for("/tmp/x/episode.mp3") == srt_path_for("/tmp/x/episode.mp3").parent / "episode.srt"
    assert srt_path_for("/tmp/x/episode.mp3", "/tmp/out") == srt_path_for("/tmp/out/episode.srt")
