import pytest

from eo_stt.srt import (
    MAX_BLOCK_CHARS,
    Segment,
    SrtWriter,
    format_block,
    format_timestamp,
    proportional_segments,
    segments_from_words,
    write_srt,
)


def test_format_timestamp():
    assert format_timestamp(0) == "00:00:00,000"
    assert format_timestamp(3661.5) == "01:01:01,500"
    assert format_timestamp(0.0004) == "00:00:00,000"
    assert format_timestamp(-5) == "00:00:00,000"


def test_proportional_segments_covers_duration():
    text = "saluton mondo kiel vi fartas hodiaŭ"
    segs = proportional_segments(text, 0.0, 10.0)
    assert segs
    assert segs[0].start == 0.0
    assert segs[-1].end == pytest.approx(10.0)
    for a, b in zip(segs, segs[1:]):
        assert b.start == pytest.approx(a.end)
    assert all(s.end > s.start for s in segs)
    assert " ".join(s.text for s in segs).split() == text.split()


def test_proportional_segments_respects_max_block_chars():
    text = " ".join(["vorto"] * 100)
    segs = proportional_segments(text, 0.0, 60.0)
    assert len(segs) > 1
    assert all(len(s.text) <= MAX_BLOCK_CHARS for s in segs)


def test_proportional_segments_empty():
    assert proportional_segments("", 0.0, 5.0) == []
    assert proportional_segments("   ", 0.0, 5.0) == []


def test_proportional_segments_min_duration():
    # last block gets a tiny share of the duration; it must still be readable
    text = " ".join(["vorto"] * 50) + " unusola"
    segs = proportional_segments(text, 0.0, 5.0)
    assert all(s.end - s.start >= 0.5 - 1e-9 for s in segs)
    assert segs[-1].end == pytest.approx(5.0)


def test_proportional_segments_offset():
    segs = proportional_segments("saluton mondo", 30.0, 4.0)
    assert segs[0].start == pytest.approx(30.0)
    assert segs[-1].end == pytest.approx(34.0)


def test_segments_from_words_breaks_on_length():
    words = [(f"w{i:02d}", i * 0.5, i * 0.5 + 0.4) for i in range(40)]
    segs = segments_from_words(words)
    assert len(segs) > 1
    assert all(len(s.text) <= MAX_BLOCK_CHARS or " " not in s.text for s in segs)
    # segments are ordered and non-overlapping
    for a, b in zip(segs, segs[1:]):
        assert b.start >= a.start


def test_segments_from_words_breaks_on_gap():
    words = [("saluton", 0.0, 0.5), ("mondo", 0.6, 1.0), ("kiel", 5.0, 5.4)]
    segs = segments_from_words(words)
    assert len(segs) == 2
    assert segs[0].text == "saluton mondo"
    assert segs[1].text == "kiel"
    assert segs[1].start == 5.0


def test_segments_from_words_breaks_on_duration():
    words = [(f"w{i}", i * 0.5, i * 0.5 + 0.4) for i in range(30)]  # 15 s of speech
    segs = segments_from_words(words)
    assert len(segs) > 1
    assert all(s.end - s.start <= 7.0 + 0.5 for s in segs)


def test_segments_from_words_min_duration():
    segs = segments_from_words([("ho", 0.0, 0.1)])
    assert segs[0].end >= 0.5


def test_write_srt_format(tmp_path):
    segs = [Segment(0.0, 1.5, "saluton"), Segment(1.5, 3.0, "mondo")]
    out = write_srt(segs, tmp_path / "x.srt")
    content = out.read_text(encoding="utf-8")
    assert content == (
        "1\n00:00:00,000 --> 00:00:01,500\nsaluton\n\n"
        "2\n00:00:01,500 --> 00:00:03,000\nmondo\n"
    )


def test_write_srt_wraps_long_lines(tmp_path):
    long_text = " ".join(["esperanto"] * 20)  # 179 chars -> wraps at 42
    out = write_srt([Segment(0.0, 10.0, long_text)], tmp_path / "x.srt")
    lines = out.read_text(encoding="utf-8").splitlines()
    text_lines = lines[2:-1]
    assert len(text_lines) > 1
    assert all(len(line) <= 42 for line in text_lines)


def test_write_srt_empty_raises(tmp_path):
    with pytest.raises(ValueError):
        write_srt([], tmp_path / "x.srt")


def test_format_block():
    assert format_block(1, Segment(0.0, 1.5, "saluton")) == [
        "1", "00:00:00,000 --> 00:00:01,500", "saluton",
    ]


def test_srt_writer_creates_file_immediately_and_appends(tmp_path):
    out = tmp_path / "x.srt"
    with SrtWriter(out) as w:
        assert out.exists()  # created up front, before any segment arrives
        assert out.read_text(encoding="utf-8") == ""
        w.append([Segment(0.0, 1.5, "saluton")])
        w.append([Segment(1.5, 3.0, "mondo"), Segment(3.0, 4.0, "kiel vi")])
    assert out.read_text(encoding="utf-8") == (
        "1\n00:00:00,000 --> 00:00:01,500\nsaluton\n\n"
        "2\n00:00:01,500 --> 00:00:03,000\nmondo\n\n"
        "3\n00:00:03,000 --> 00:00:04,000\nkiel vi\n\n"
    )


def test_srt_writer_blocks_match_write_srt(tmp_path):
    # the incremental writer and write_srt must produce the same blocks
    segs = [Segment(0.0, 1.5, "saluton"), Segment(1.5, 3.0, "mondo")]
    incremental = tmp_path / "inc.srt"
    with SrtWriter(incremental) as w:
        w.append(segs[:1])
        w.append(segs[1:])
    final = tmp_path / "final.srt"
    write_srt(segs, final)
    inc_text = incremental.read_text(encoding="utf-8")
    final_text = final.read_text(encoding="utf-8")
    assert inc_text.rstrip("\n") == final_text.rstrip("\n")
