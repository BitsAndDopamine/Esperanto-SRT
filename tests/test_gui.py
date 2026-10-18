"""Tests for the GUI's file-selector helpers (headless — subprocess/dialogs mocked).

The App-level tests at the bottom instantiate a real Tk app and are skipped
when no display is available (CI runs pytest headless).
"""

import os
import subprocess
import sys
import time
import tkinter as tk
from pathlib import Path
from unittest import mock

import pytest

import eo_stt.gui as gui
from eo_stt import __version__


def _display_available() -> bool:
    try:
        root = tk.Tk()
    except tk.TclError:
        return False
    root.destroy()
    return True


needs_display = pytest.mark.skipif(not _display_available(), reason="needs a display")


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


# -- file-type dropdown ------------------------------------------------------


def test_audio_types_list_individual_formats():
    # the combined "audio files" entry is preselected, each supported format is
    # selectable on its own, and "all files" is last
    labels = [label for label, _ in gui.AUDIO_TYPES]
    patterns = [pats for _, pats in gui.AUDIO_TYPES]
    assert labels[0] == "aŭdiaj dosieroj"
    assert patterns[0] == "*.mp3 *.wav *.ogg *.flac"
    for ext in ("mp3", "wav", "ogg", "flac"):
        assert f"*.{ext}" in patterns
    assert patterns[-1] == "*"


def test_zenity_receives_per_format_filters():
    captured = {}

    def run(cmd, **kwargs):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    with mock.patch.object(gui, "_zenity_path", return_value="/usr/bin/zenity"), \
         mock.patch.object(gui.subprocess, "run", run):
        gui._ask_open_filenames("t")
    filters = [a for a in captured["cmd"] if a.startswith("--file-filter=")]
    assert "--file-filter=aŭdiaj dosieroj | *.mp3 *.wav *.ogg *.flac" in filters
    assert "--file-filter=mp3-dosieroj (*.mp3) | *.mp3" in filters
    assert "--file-filter=wav-dosieroj (*.wav) | *.wav" in filters
    assert "--file-filter=ogg-dosieroj (*.ogg) | *.ogg" in filters
    assert "--file-filter=flac-dosieroj (*.flac) | *.flac" in filters
    # GTK already offers its own "all files" filter; no duplicate needed
    assert not any(f.endswith("| *") for f in filters)


def test_tk_dialog_gets_per_format_filetypes():
    with mock.patch.object(gui.shutil, "which", return_value=None), \
         mock.patch.object(gui.filedialog, "askopenfilenames", return_value=()) as dlg:
        gui._ask_open_filenames("t")
    filetypes = dlg.call_args.kwargs["filetypes"]
    assert filetypes == gui.AUDIO_TYPES
    assert filetypes[0][0] == "aŭdiaj dosieroj"  # preselected entry
    assert "*.mp3" in [pats for _, pats in filetypes]
    assert "*.flac" in [pats for _, pats in filetypes]


# -- progress bar + status line (need a real Tk app) --------------------------


def _drain(app, timeout: float = 10.0) -> None:
    """Process Tk events until the worker thread is done and the queue is empty."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.update()
        worker_done = app._worker is None or not app._worker.is_alive()
        if worker_done and app._events.empty():
            time.sleep(0.15)  # let the after(100) _poll cycle run once more
            app.update()
            if app._events.empty():
                return
        time.sleep(0.01)
    raise AssertionError("timed out waiting for the GUI worker")


@needs_display
def test_run_shows_immediate_feedback_then_progress(tmp_path, monkeypatch):
    opened = []
    events = []

    def fake_transcribe_file(model, path, out_dir, on_progress=None):
        if on_progress is not None:
            on_progress(0, 2)
            on_progress(1, 2)
            on_progress(2, 2)
        out = gui.srt_path_for(path, out_dir)
        out.write_text("1\n00:00:00,000 --> 00:00:01,000\nsaluton\n", encoding="utf-8")
        return out

    monkeypatch.setattr(gui, "load_model", lambda: object())
    monkeypatch.setattr(gui, "transcribe_file", fake_transcribe_file)

    app = gui.App()
    try:
        monkeypatch.setattr(app, "_open_preview", lambda p: opened.append(p))
        orig_put = app._events.put

        def spy_put(item, *args, **kwargs):
            events.append(item)
            return orig_put(item, *args, **kwargs)

        monkeypatch.setattr(app._events, "put", spy_put)
        wav = tmp_path / "a.wav"
        wav.write_bytes(b"RIFF....")  # content irrelevant, transcribe_file is faked
        app._files.append(wav)
        app._out_var.set(str(tmp_path))

        app._start()

        # immediate feedback, synchronously after clicking "transskribu":
        # busy animation on the bar, status line, disabled button
        assert app._status.cget("text") == "ŝargante modelon..."
        assert app._progress._loop_running
        assert str(app._run_btn.cget("state")) == "disabled"

        _drain(app)

        # model loading is its own first step, before any file progress
        kinds = [kind for kind, _ in events]
        assert "model_loaded" in kinds
        assert kinds.index("model_loaded") < kinds.index("done")
        model_loaded = next(p for k, p in events if k == "model_loaded")
        assert model_loaded == gui.MODEL_LOAD_SHARE * 100

        # progress values only ever move forward, ending at 100%
        progress = [p for k, p in events if k == "progress"]
        assert progress == sorted(progress)
        assert progress[-1] == 100

        assert app._progress.get() == 1.0
        assert app._progress._loop_running is False
        assert app._status.cget("text") == "farita."
        assert str(app._run_btn.cget("state")) == "normal"
        assert opened == [tmp_path / "a.srt"]
    finally:
        app.destroy()


@needs_display
def test_run_failure_stops_animation_and_reports(tmp_path, monkeypatch):
    errors = []

    def boom():
        raise RuntimeError("modelo mankas")

    monkeypatch.setattr(gui, "load_model", boom)
    monkeypatch.setattr(gui.messagebox, "showerror", lambda *a: errors.append(a))

    app = gui.App()
    try:
        wav = tmp_path / "a.wav"
        wav.write_bytes(b"RIFF....")
        app._files.append(wav)

        app._start()
        assert app._progress._loop_running  # busy animation while loading

        _drain(app)

        assert app._progress._loop_running is False
        assert app._status.cget("text") == "eraro"
        assert str(app._run_btn.cget("state")) == "normal"
        assert errors and "modelo mankas" in errors[0][1]
    finally:
        app.destroy()


# -- "pri" (about) window ------------------------------------------------------


def test_about_text_mentions_version_repo_and_licenses():
    text = gui.ABOUT_TEXT
    assert f"v{__version__}" in text
    assert "https://github.com/BitsAndDopamine/Esperanto-STT" in text
    assert "CC0 1.0" in text    # license of the code
    assert "CC-BY-4.0" in text  # license of the bundled model


@needs_display
def test_about_window_opens():
    app = gui.App()
    try:
        app._show_about()
        app.update()
        windows = [w for w in app.winfo_children() if isinstance(w, gui.AboutWindow)]
        assert len(windows) == 1
        assert windows[0].title() == "pri eo-stt"
        windows[0].destroy()
    finally:
        app.destroy()


# -- "srt" button enabled only when the .srt exists ----------------------------


@needs_display
def test_srt_button_only_enabled_when_srt_exists(tmp_path, monkeypatch):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF....")
    monkeypatch.setattr(gui, "_ask_open_filenames", lambda title: [wav])

    app = gui.App()
    try:
        app._add_files()
        btn = app._srt_buttons[wav]
        # no .srt yet (the default output folder is the file's own folder)
        assert str(btn.cget("state")) == "disabled"

        srt = tmp_path / "a.srt"
        srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nsaluton\n", encoding="utf-8")
        app._refresh_srt_buttons()
        assert str(btn.cget("state")) == "normal"

        # a different output folder has no .srt for this file -> disabled again
        other = tmp_path / "alia"
        other.mkdir()
        app._out_var.set(str(other))
        app._refresh_srt_buttons()
        assert str(btn.cget("state")) == "disabled"
    finally:
        app.destroy()


@needs_display
def test_srt_button_enabled_after_run(tmp_path, monkeypatch):
    def fake_transcribe_file(model, path, out_dir, on_progress=None):
        out = gui.srt_path_for(path, out_dir)
        out.write_text("1\n00:00:00,000 --> 00:00:01,000\nsaluton\n", encoding="utf-8")
        return out

    monkeypatch.setattr(gui, "load_model", lambda: object())
    monkeypatch.setattr(gui, "transcribe_file", fake_transcribe_file)
    monkeypatch.setattr(gui, "_ask_open_filenames", lambda title: [tmp_path / "a.wav"])
    (tmp_path / "a.wav").write_bytes(b"RIFF....")

    app = gui.App()
    try:
        monkeypatch.setattr(app, "_open_preview", lambda p: None)
        app._add_files()
        btn = app._srt_buttons[tmp_path / "a.wav"]
        assert str(btn.cget("state")) == "disabled"

        app._start()
        _drain(app)

        # the run wrote the .srt; the worker events refreshed the button
        assert str(btn.cget("state")) == "normal"
    finally:
        app.destroy()
