"""Tests for the GUI's file-selector helpers (headless — subprocess/dialogs mocked)."""

import os
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from unittest import mock

import pytest

import eo_stt.gui as gui


def _fake_zenity(stdout="", returncode=0, stderr=""):
    def run(cmd, **kwargs):
        assert cmd[0].endswith("zenity")
        return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr=stderr)

    return run


def test_zenity_multiple_files_parsing():
    with mock.patch.object(gui, "_zenity_path", return_value="/usr/bin/zenity"), \
         mock.patch.object(gui.subprocess, "run", _fake_zenity(stdout="/a/b.mp3\n/c/d.wav\n")):
        assert gui._ask_open_filenames("t") == [Path("/a/b.mp3"), Path("/c/d.wav")]


def test_zenity_cancel_returns_empty():
    with mock.patch.object(gui, "_zenity_path", return_value="/usr/bin/zenity"), \
         mock.patch.object(gui.subprocess, "run", _fake_zenity(stdout="", returncode=1)):
        assert gui._ask_open_filenames("t") == []


def test_zenity_directory():
    with mock.patch.object(gui, "_zenity_path", return_value="/usr/bin/zenity"), \
         mock.patch.object(gui.subprocess, "run", _fake_zenity(stdout="/some/dir\n")):
        assert gui._ask_directory("t") == "/some/dir"


def test_zenity_directory_cancel_returns_empty():
    with mock.patch.object(gui, "_zenity_path", return_value="/usr/bin/zenity"), \
         mock.patch.object(gui.subprocess, "run", _fake_zenity(stdout="", returncode=1)):
        assert gui._ask_directory("t") == ""


def test_fallback_to_tkinter_files_when_no_zenity():
    with mock.patch.object(gui.shutil, "which", return_value=None), \
         mock.patch.object(gui.filedialog, "askopenfilenames", return_value=("/x.mp3",)) as dlg:
        assert gui._ask_open_filenames("t") == [Path("/x.mp3")]
        dlg.assert_called_once()


def test_fallback_to_tkinter_directory_when_no_zenity():
    with mock.patch.object(gui.shutil, "which", return_value=None), \
         mock.patch.object(gui.filedialog, "askdirectory", return_value="/out") as dlg:
        assert gui._ask_directory("t") == "/out"
        dlg.assert_called_once()


def test_no_zenity_on_windows_macos():
    # on Windows/macOS tkinter's filedialog is already the native OS dialog
    with mock.patch.object(gui.sys, "platform", "win32"), \
         mock.patch.object(gui.filedialog, "askopenfilenames", return_value=()) as dlg, \
         mock.patch.object(gui.subprocess, "run") as run:
        gui._ask_open_filenames("t")
        dlg.assert_called_once()
        run.assert_not_called()


def test_zenity_failure_falls_back_to_tkinter():
    # a crashing zenity (e.g. exit 127 = loader error) is a failure, not a cancel
    with mock.patch.object(gui, "_zenity_path", return_value="/usr/bin/zenity"), \
         mock.patch.object(gui.subprocess, "run", _fake_zenity(returncode=127, stderr="symbol lookup error")), \
         mock.patch.object(gui.filedialog, "askopenfilenames", return_value=("/x.mp3",)) as dlg:
        assert gui._ask_open_filenames("t") == [Path("/x.mp3")]
        dlg.assert_called_once()


def test_zenity_directory_failure_falls_back_to_tkinter():
    with mock.patch.object(gui, "_zenity_path", return_value="/usr/bin/zenity"), \
         mock.patch.object(gui.subprocess, "run", _fake_zenity(returncode=127)), \
         mock.patch.object(gui.filedialog, "askdirectory", return_value="/out") as dlg:
        assert gui._ask_directory("t") == "/out"
        dlg.assert_called_once()


def test_zenity_oserror_falls_back_to_tkinter():
    def boom(cmd, **kwargs):
        raise OSError("exec failed")

    with mock.patch.object(gui, "_zenity_path", return_value="/usr/bin/zenity"), \
         mock.patch.object(gui.subprocess, "run", boom), \
         mock.patch.object(gui.filedialog, "askopenfilenames", return_value=()) as dlg:
        assert gui._ask_open_filenames("t") == []
        dlg.assert_called_once()


def test_tk_dialog_error_is_reported_not_silent():
    with mock.patch.object(gui.shutil, "which", return_value=None), \
         mock.patch.object(gui.filedialog, "askopenfilenames", side_effect=tk.TclError("no display")), \
         mock.patch.object(gui.messagebox, "showerror") as err:
        assert gui._ask_open_filenames("t") == []
        err.assert_called_once()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="zenity is Linux-only")
def test_zenity_child_gets_clean_env(tmp_path, monkeypatch):
    """Regression test for the frozen build: the PyInstaller bootloader points
    LD_LIBRARY_PATH at the bundled _internal/ libraries; a zenity child must not
    inherit it (the bundled system libraries can crash GTK tools with symbol
    lookup errors — that is what silently broke the file dialogs in the v0.1.0
    Linux artifact)."""
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    dump = tmp_path / "zenity-env.txt"
    fake = fakebin / "zenity"
    fake.write_text(f'#!/bin/sh\nenv > "{dump}"\nprintf "/a/b.mp3\\n"\n')
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fakebin}:{os.environ.get('PATH', '')}")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/frozen/_internal")
    monkeypatch.setenv("LD_PRELOAD", "/frozen/libevil.so")
    assert gui._ask_open_filenames("t") == [Path("/a/b.mp3")]
    child_env = dump.read_text(encoding="utf-8")
    assert "LD_LIBRARY_PATH" not in child_env
    assert "LD_PRELOAD" not in child_env
    assert "PATH=" in child_env  # the rest of the environment is preserved
