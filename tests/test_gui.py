"""Tests for the GUI's file-selector helpers (headless — subprocess/dialogs mocked)."""

import subprocess
from pathlib import Path
from unittest import mock

import eo_stt.gui as gui


def _fake_zenity(stdout="", returncode=0):
    def run(cmd, **kwargs):
        assert cmd[0].endswith("zenity")
        return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr="")

    return run


def test_zenity_multiple_files_parsing():
    with mock.patch.object(gui.shutil, "which", return_value="/usr/bin/zenity"), \
         mock.patch.object(gui.subprocess, "run", _fake_zenity(stdout="/a/b.mp3\n/c/d.wav\n")):
        assert gui._ask_open_filenames("t") == [Path("/a/b.mp3"), Path("/c/d.wav")]


def test_zenity_cancel_returns_empty():
    with mock.patch.object(gui.shutil, "which", return_value="/usr/bin/zenity"), \
         mock.patch.object(gui.subprocess, "run", _fake_zenity(stdout="", returncode=1)):
        assert gui._ask_open_filenames("t") == []


def test_zenity_directory():
    with mock.patch.object(gui.shutil, "which", return_value="/usr/bin/zenity"), \
         mock.patch.object(gui.subprocess, "run", _fake_zenity(stdout="/some/dir\n")):
        assert gui._ask_directory("t") == "/some/dir"


def test_zenity_directory_cancel_returns_empty():
    with mock.patch.object(gui.shutil, "which", return_value="/usr/bin/zenity"), \
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
