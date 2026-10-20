"""Guard for user-facing UI strings.

- cli.py must stay pure ASCII: Windows consoles can fail to encode non-ASCII
  prints.
- gui.py is Esperanto with proper diacritics (ĉĝĥĵŝŭ and capitals). Tk's font
  fallback renders them (CTk widgets via the theme font, plain-Tk dialogs via
  the reconfigured named fonts — see _default_font_family/_setup_tk_fonts in
  gui.py), but anything beyond the Esperanto letters could garble, so only
  those are allowed.

Transcript text itself is Esperanto and lives only in the .srt files (UTF-8),
never in widgets.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLI_SOURCE = ROOT / "eo_srt" / "cli.py"
GUI_SOURCE = ROOT / "eo_srt" / "gui.py"

# the only non-ASCII characters allowed in gui.py
ESPERANTO_LETTERS = set("ĉĝĥĵŝŭĈĜĤĴŜŬ")


def test_cli_source_is_ascii():
    text = CLI_SOURCE.read_text(encoding="utf-8")
    non_ascii = sorted({ch for ch in text if ord(ch) > 127})
    assert not non_ascii, f"cli.py contains non-ASCII characters: {non_ascii!r}"


def test_gui_source_only_esperanto_diacritics():
    text = GUI_SOURCE.read_text(encoding="utf-8")
    bad = sorted({ch for ch in text if ord(ch) > 127 and ch not in ESPERANTO_LETTERS})
    assert not bad, (
        f"gui.py contains non-ASCII characters beyond the Esperanto diacritics "
        f"{sorted(ESPERANTO_LETTERS)!r}: {bad!r}"
    )
