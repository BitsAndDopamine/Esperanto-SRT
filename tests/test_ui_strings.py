"""Guard against non-ASCII UI strings.

Tk's default font is not guaranteed to have non-ASCII glyphs (on this machine
it is Latin-1-only "Nimbus Sans L", which garbles …, →, — and the Esperanto
diacritics), and Windows consoles can fail to encode them. All user-facing
strings in gui.py/cli.py must therefore stay ASCII. Transcript text itself is
Esperanto and lives only in the .srt files (UTF-8), never in widgets.
"""

from pathlib import Path

import pytest

UI_SOURCES = [
    Path(__file__).resolve().parent.parent / "eo_stt" / "gui.py",
    Path(__file__).resolve().parent.parent / "eo_stt" / "cli.py",
]


@pytest.mark.parametrize("path", UI_SOURCES, ids=lambda p: p.name)
def test_ui_source_is_ascii(path):
    text = path.read_text(encoding="utf-8")
    non_ascii = sorted({ch for ch in text if ord(ch) > 127})
    assert not non_ascii, f"{path.name} contains non-ASCII characters: {non_ascii!r}"
