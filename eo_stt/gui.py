"""CustomTkinter GUI (Linux + Windows + macOS, dark mode).

Run with: python -m eo_stt   (or the PyInstaller executable with no arguments)

All transcription happens in a worker thread; the GUI only receives events
through a queue (Tk is not thread-safe). All widget/log strings are ASCII,
enforced by tests/test_ui_strings.py (Tk/console fonts are not guaranteed to
have non-ASCII glyphs).
"""

from __future__ import annotations

import logging
import queue
import shutil
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk

from .model import MODEL_ID, load_model
from .transcribe import transcribe_file

AUDIO_TYPES = [("Audio files", "*.mp3 *.wav *.ogg *.flac"), ("All files", "*.*")]


def _zenity_path() -> str | None:
    """Path to zenity (the native GNOME file selector) on Linux, else None."""
    if sys.platform.startswith("linux"):
        return shutil.which("zenity")
    return None


def _ask_open_filenames(title: str) -> list[Path]:
    """Pick audio files with the OS-native selector.

    Linux/GNOME: zenity (the native GTK file dialog) when installed.
    Windows/macOS: tkinter's filedialog (already the native OS dialog).
    Linux without zenity: tkinter's Tk dialog (fallback).
    """
    zenity = _zenity_path()
    if zenity:
        proc = subprocess.run(
            [
                zenity, "--file-selection", "--multiple",
                "--separator", "\n",
                f"--title={title}",
                "--file-filter=Audio files | *.mp3 *.wav *.ogg *.flac",
            ],
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            return [Path(p) for p in proc.stdout.splitlines() if p.strip()]
        return []  # user cancelled
    paths = filedialog.askopenfilenames(title=title, filetypes=AUDIO_TYPES)
    return [Path(p) for p in paths]


def _ask_directory(title: str) -> str:
    """Pick a folder with the OS-native selector (same fallback chain)."""
    zenity = _zenity_path()
    if zenity:
        proc = subprocess.run(
            [zenity, "--file-selection", "--directory", f"--title={title}"],
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            return proc.stdout.strip()
        return ""  # user cancelled
    return filedialog.askdirectory(title=title)


class _QueueHandler(logging.Handler):
    """Forward eo_stt log records into the GUI event queue (worker thread safe)."""

    def __init__(self, events: queue.Queue) -> None:
        super().__init__()
        self._events = events

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._events.put(("log", self.format(record)))
        except Exception:
            pass


def _default_font_family(root: tk.Misc) -> str:
    """Pick a font family the OS definitely has, for all CustomTkinter widgets.

    CTk's default family is its bundled "Roboto", which on Linux is only copied
    to ~/.fonts (fontconfig may not see it without a cache refresh) and is not
    installed on macOS at all; Tk then falls back to an ugly bitmap font. All
    candidates below cover the Esperanto diacritics (c^, g^, h^, j^, s^, u^).
    """
    from tkinter import font as tkfont

    families = set(tkfont.families(root))
    for family in ("DejaVu Sans", "Noto Sans", "Liberation Sans", "Segoe UI", "Helvetica", "Roboto", "Arial"):
        if family in families:
            return family
    return "Roboto"  # CTk's own default


class App(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("eo-stt - Esperanto speech-to-text (mp3 -> srt)")
        self.geometry("760x560")
        self.minsize(680, 460)
        ctk.ThemeManager.theme["CTkFont"]["family"] = _default_font_family(self)
        self._events: queue.Queue = queue.Queue()
        self._worker: threading.Thread | None = None
        self._files: list[Path] = []
        self._rows: dict[Path, ctk.CTkFrame] = {}
        logger = logging.getLogger("eo_stt")
        logger.setLevel(logging.INFO)
        logger.addHandler(_QueueHandler(self._events))
        self._build()
        self.after(100, self._poll)

    def _build(self) -> None:
        pad = {"padx": 10, "pady": 6}
        # corner_radius=0 everywhere: on some systems (this machine's
        # Linux/Xwayland + Tk) CTk's rounded corners render as jagged notches.
        # Sharp rectangles render cleanly.
        cr = {"corner_radius": 0}

        top = ctk.CTkFrame(self, fg_color="transparent", **cr)
        top.pack(fill="both", expand=True, **pad)
        self._list_frame = ctk.CTkScrollableFrame(top, **cr)
        self._list_frame.pack(side="left", fill="both", expand=True)
        self._placeholder = ctk.CTkLabel(self._list_frame, text="(no files yet)", anchor="w")
        self._placeholder.pack(fill="x", pady=2)
        btns = ctk.CTkFrame(top, fg_color="transparent", **cr)
        btns.pack(side="left", fill="y", padx=(8, 0))
        ctk.CTkButton(btns, text="Add files...", command=self._add_files, **cr).pack(fill="x", pady=2)
        ctk.CTkButton(btns, text="Clear", command=self._clear_files, **cr).pack(fill="x", pady=2)

        out = ctk.CTkFrame(self, fg_color="transparent", **cr)
        out.pack(fill="x", **pad)
        ctk.CTkLabel(out, text="Output folder:").pack(side="left")
        self._out_var = tk.StringVar()
        ctk.CTkEntry(out, textvariable=self._out_var, **cr).pack(side="left", fill="x", expand=True, padx=6)
        ctk.CTkButton(out, text="Browse...", command=self._browse_out, **cr).pack(side="left")

        run = ctk.CTkFrame(self, fg_color="transparent", **cr)
        run.pack(fill="x", **pad)
        self._run_btn = ctk.CTkButton(run, text="Transcribe -> .srt", command=self._start, **cr)
        self._run_btn.pack(side="left")
        self._progress = ctk.CTkProgressBar(run, **cr)
        self._progress.pack(side="left", fill="x", expand=True, padx=10)
        self._progress.set(0)

        self._log = ctk.CTkTextbox(self, state="disabled", **cr)
        self._log.pack(fill="both", expand=True, **pad)

    # -- widget helpers ---------------------------------------------------

    def _append_log(self, msg: str) -> None:
        self._log.configure(state="normal")
        self._log.insert("end", msg + "\n")
        self._log.see("end")
        self._log.configure(state="disabled")

    def _add_files(self) -> None:
        for path in _ask_open_filenames("Choose audio files"):
            if path in self._rows:
                continue
            row = ctk.CTkFrame(self._list_frame, fg_color="transparent", corner_radius=0)
            row.pack(fill="x", pady=1)
            ctk.CTkLabel(row, text=path.name, anchor="w").pack(side="left", fill="x", expand=True)
            ctk.CTkButton(
                row, text="x", width=28, height=28, corner_radius=0,
                command=lambda p=path: self._remove_file(p),
            ).pack(side="right", padx=2)
            self._rows[path] = row
            self._files.append(path)
        if self._files:
            self._placeholder.pack_forget()
        if self._files and not self._out_var.get():
            self._out_var.set(str(self._files[0].parent))

    def _remove_file(self, path: Path) -> None:
        row = self._rows.pop(path, None)
        if row is not None:
            row.destroy()
        if path in self._files:
            self._files.remove(path)
        if not self._files:
            self._placeholder.pack(fill="x", pady=2)

    def _clear_files(self) -> None:
        for row in self._rows.values():
            row.destroy()
        self._rows.clear()
        self._files.clear()
        self._placeholder.pack(fill="x", pady=2)

    def _browse_out(self) -> None:
        directory = _ask_directory("Choose output folder")
        if directory:
            self._out_var.set(directory)

    # -- transcription ----------------------------------------------------

    def _start(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        if not self._files:
            messagebox.showinfo("eo-stt", "Add at least one audio file first.")
            return
        self._run_btn.configure(state="disabled")
        self._progress.set(0)
        self._worker = threading.Thread(target=self._work, daemon=True)
        self._worker.start()

    def _work(self) -> None:
        files = list(self._files)
        out_dir = self._out_var.get().strip() or None
        n_files = len(files)
        try:
            self._events.put(("log", f"Loading model {MODEL_ID} (first run downloads ~500 MB)..."))
            model = load_model()
            for i, f in enumerate(files, 1):
                self._events.put(("progress", (i - 1) / n_files * 100))
                self._events.put(("log", f"[{i}/{n_files}] {f.name} ..."))
                base = (i - 1) / n_files * 100
                step = 100 / n_files

                def on_progress(done: int, total: int, base: float = base, step: float = step) -> None:
                    self._events.put(("progress", base + step * done / total))

                out = transcribe_file(model, f, out_dir, on_progress=on_progress)
                self._events.put(("log", f"    wrote {out}"))
            self._events.put(("progress", 100))
            self._events.put(("log", "Done."))
            self._events.put(("done", None))
        except Exception as exc:  # show any failure in the GUI
            self._events.put(("error", str(exc)))
        finally:
            self._events.put(("finished", None))

    def _poll(self) -> None:
        try:
            while True:
                kind, payload = self._events.get_nowait()
                if kind == "log":
                    self._append_log(payload)
                elif kind == "progress":
                    self._progress.set(payload / 100)
                elif kind == "done":
                    messagebox.showinfo("eo-stt", "All files transcribed.")
                elif kind == "error":
                    messagebox.showerror("eo-stt - error", payload)
                elif kind == "finished":
                    self._run_btn.configure(state="normal")
        except queue.Empty:
            pass
        self.after(100, self._poll)


def main() -> None:
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    App().mainloop()


if __name__ == "__main__":
    main()
