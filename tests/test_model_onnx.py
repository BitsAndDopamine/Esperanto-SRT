"""Tests for the ONNX backend glue in eo_srt.model (no model download needed
for the unit tests; the integration test skips when the local INT8 model files
are not present)."""

import numpy as np
import pytest

from eo_srt.model import (
    MODEL_DIR_NAME,
    OnnxAsrModel,
    _words_from_tokens,
    default_model_dir,
    load_model,
)


def test_words_from_tokens_groups_by_space():
    tokens = [" sal", "ut", "on", " kaj", " mond", "o"]
    timestamps = [0.0, 0.4, 0.8, 1.2, 1.6, 2.0]
    words = _words_from_tokens(tokens, timestamps)
    assert [w for w, _, _ in words] == ["saluton", "kaj", "mondo"]
    # token spans are [t_i, t_{i+1}); the last token gets one extra frame
    assert words[0] == ("saluton", 0.0, 1.2)
    assert words[1] == ("kaj", 1.2, 1.6)
    assert words[2] == ("mondo", 1.6, 2.04)


def test_words_from_tokens_same_frame_tokens_share_span():
    # several tokens emitted at the same encoder frame (max_tokens_per_step)
    tokens = [" a", "b", " c"]
    timestamps = [1.0, 1.0, 1.0]
    words = _words_from_tokens(tokens, timestamps)
    assert words[0] == ("ab", 1.0, 1.0)
    assert words[1] == ("c", 1.0, 1.04)


def test_words_from_tokens_no_leading_space_first_token():
    words = _words_from_tokens(["saluton", " mondo"], [0.0, 1.0])
    assert [w for w, _, _ in words] == ["saluton", "mondo"]


def test_words_from_tokens_empty():
    assert _words_from_tokens([], []) == []


def test_default_model_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("EO_SRT_MODEL_DIR", str(tmp_path))
    assert default_model_dir() == tmp_path


def test_default_model_dir_source_tree():
    # running from source: <repo>/models/<MODEL_DIR_NAME>
    d = default_model_dir()
    assert d.name == MODEL_DIR_NAME
    assert d.parent.name == "models"


def test_load_model_missing_dir_raises(tmp_path):
    with pytest.raises(RuntimeError, match="Cannot load the ASR model"):
        load_model(tmp_path / "does-not-exist")


@pytest.mark.skipif(
    not (default_model_dir() / "config.json").is_file(),
    reason="local INT8 model files not present (run: python fetch_model.py)",
)
def test_onnx_model_transcribes_silence(tmp_path):
    """Integration smoke test with the real INT8 model: 2 s of silence."""
    import soundfile as sf

    wav = tmp_path / "silence.wav"
    sf.write(str(wav), np.zeros(2 * 16000, dtype=np.float32), 16000)

    model = load_model()
    assert isinstance(model, OnnxAsrModel)
    from eo_srt.audio import load_audio
    from eo_srt.transcribe import transcribe_file

    samples = load_audio(wav)
    hyps = model.transcribe([samples])
    assert len(hyps) == 1
    assert isinstance(hyps[0].text, str)
    out = transcribe_file(model, wav)
    assert out.is_file()
