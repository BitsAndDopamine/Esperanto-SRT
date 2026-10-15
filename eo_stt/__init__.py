"""eo-stt — Esperanto speech-to-text (mp3 -> srt) with a GUI."""

from __future__ import annotations

import os

__version__ = "0.1.0"

# Dynamic-loader variables. In a PyInstaller build the bootloader points
# LD_LIBRARY_PATH at the bundled _internal/ libraries (a setenv after startup,
# so it is invisible in /proc/<pid>/environ but inherited by child processes).
# Children must NOT inherit them: system tools spawned by the app (the zenity
# file dialogs, the ffmpeg fallback) would load the bundled system libraries
# instead of the system ones and can crash — e.g. the CI build (python.org
# Python, Tk 8.6) bundles an older libfontconfig, and zenity died with
# "symbol lookup error: libpangoft2: undefined symbol: FcConfigSetDefaultSubstitute".
_LOADER_VARS = (
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "DYLD_LIBRARY_PATH",
    "DYLD_FALLBACK_LIBRARY_PATH",
)


def subprocess_env() -> dict[str, str]:
    """os.environ without dynamic-loader overrides, for spawning subprocesses.

    Everything else (PATH, DISPLAY, HOME, locale, ...) is preserved, so
    system tools like zenity/ffmpeg behave exactly as if started from a
    terminal. See _ask_open_filenames/_ask_directory in gui.py and
    _read_with_ffmpeg in audio.py.
    """
    env = os.environ.copy()
    for var in _LOADER_VARS:
        env.pop(var, None)
    return env
