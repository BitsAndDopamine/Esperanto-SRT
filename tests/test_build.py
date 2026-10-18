"""Tests for build.py's executable naming (no PyInstaller run needed).

The Linux executable name must not end in ".<digit>": the shared MIME
database maps "*.1".."*.9" to man pages (application/x-troff-man, a text
type), so a file named e.g. "eo-stt-v0.2" is opened as a text file by Linux
file managers on double-click instead of being run. build.py therefore
appends "-linux" (Windows keeps the ".exe" extension, which has no such
rule). Verified against the real MIME database with:
    xdg-mime query filetype <file>
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import build  # noqa: E402
from eo_stt import __version__  # noqa: E402


def test_linux_exe_name_has_version_but_no_trailing_digit():
    name = build.versioned_exe_name("eo-stt", "v0.2", "linux")
    assert name == "eo-stt-v0.2-linux"
    assert name.endswith("-linux")


def test_windows_exe_name():
    assert build.versioned_exe_name("eo-stt", "v0.2", "win32") == "eo-stt-v0.2.exe"


def test_exe_name_never_ends_in_digit():
    # a trailing digit would hit the "*.1".."*.9" man-page globs (Linux)
    for platform in ("linux", "win32", "darwin"):
        name = build.versioned_exe_name("eo-stt", "v0.2", platform)
        assert not name[-1].isdigit(), f"{platform}: {name} ends in a digit"


def test_version_label_defaults_to_package_version():
    assert build.versioned_exe_name("eo-stt", None, "linux") == f"eo-stt-v{__version__}-linux"
    assert build.versioned_exe_name("eo-stt", None, "win32") == f"eo-stt-v{__version__}.exe"
