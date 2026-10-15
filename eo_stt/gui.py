"""CustomTkinter GUI (Linux + Windows + macOS, dark mode).

Run with: python -m eo_stt   (or the PyInstaller executable with no arguments)

All transcription happens in a worker thread; the GUI only receives events
through a queue (Tk is not thread-safe). The interface language is Esperanto
with proper diacritics (ĉĝĥĵŝŭ). CTk widgets get a Unicode-capable font family
via the theme and Tk's own named fonts are reconfigured the same way (see
_default_font_family/_setup_tk_fonts), so the diacritics render in widgets
and in plain-Tk dialogs (messagebox, the Tk file-dialog fallback).
tests/test_ui_strings.py enforces that gui.py uses nothing beyond the
Esperanto diacritics, and that cli.py stays pure ASCII (Windows consoles can
fail to encode non-ASCII prints).
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

from . import subprocess_env
from .model import load_model
from .transcribe import srt_path_for, transcribe_file

AUDIO_TYPES = [("aŭdiaj dosieroj", "*.mp3 *.wav *.ogg *.flac"), ("ĉiuj dosieroj", "*.*")]


def _zenity_path() -> str | None:
    """Path to zenity (the native GNOME file selector) on Linux, else None."""
    if sys.platform.startswith("linux"):
        return shutil.which("zenity")
    return None


def _run_zenity(zenity: str, args: list[str]) -> subprocess.CompletedProcess | None:
    """Run zenity with a clean environment (see eo_stt.subprocess_env).

    Returns None when zenity could not be started at all (OSError).
    """
    try:
        return subprocess.run(
            [zenity, *args],
            capture_output=True,
            text=True,
            env=subprocess_env(),
        )
    except OSError as exc:
        logging.getLogger("eo_stt").warning("zenity ne povis starti: %s", exc)
        return None


def _ask_open_filenames(title: str) -> list[Path]:
    """Pick audio files with the OS-native selector.

    Linux/GNOME: zenity (the native GTK file dialog) when installed.
    Windows/macOS: tkinter's filedialog (already the native OS dialog).
    Linux without zenity (or when zenity fails): tkinter's Tk dialog.
    """
    zenity = _zenity_path()
    if zenity:
        proc = _run_zenity(
            zenity,
            [
                "--file-selection", "--multiple",
                "--separator", "\n",
                f"--title={title}",
                "--file-filter=aŭdiaj dosieroj | *.mp3 *.wav *.ogg *.flac",
            ],
        )
        if proc is not None:
            if proc.returncode == 0:
                return [Path(p) for p in proc.stdout.splitlines() if p.strip()]
            if proc.returncode == 1:
                return []  # user cancelled
            # any other exit code is a failure, not a cancel: say so and
            # fall back to the Tk dialog instead of silently doing nothing
            logging.getLogger("eo_stt").warning(
                "zenity malsukcesis (elirkodo %s): %s",
                proc.returncode, proc.stderr.strip() or "neniu detalo",
            )
    try:
        paths = filedialog.askopenfilenames(title=title, filetypes=AUDIO_TYPES)
    except tk.TclError as exc:
        messagebox.showerror("eo-stt - eraro", f"ne eblas malfermi la dosier-elektilon:\n{exc}")
        return []
    return [Path(p) for p in paths]


def _ask_directory(title: str) -> str:
    """Pick a folder with the OS-native selector (same fallback chain)."""
    zenity = _zenity_path()
    if zenity:
        proc = _run_zenity(zenity, ["--file-selection", "--directory", f"--title={title}"])
        if proc is not None:
            if proc.returncode == 0:
                return proc.stdout.strip()
            if proc.returncode == 1:
                return ""  # user cancelled
            logging.getLogger("eo_stt").warning(
                "zenity malsukcesis (elirkodo %s): %s",
                proc.returncode, proc.stderr.strip() or "neniu detalo",
            )
    try:
        return filedialog.askdirectory(title=title)
    except tk.TclError as exc:
        messagebox.showerror("eo-stt - eraro", f"ne eblas malfermi la dosieruj-elektilon:\n{exc}")
        return ""


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


def _setup_tk_fonts(root: tk.Misc, family: str) -> None:
    """Point Tk's own named fonts at a Unicode-capable family.

    CTk widgets get the family via the theme (see _default_font_family), but
    plain-Tk dialogs (messagebox, the Tk file-dialog fallback) use Tk's named
    fonts; on some systems those are Latin-1-only and would garble the
    Esperanto diacritics in message texts.
    """
    from tkinter import font as tkfont

    for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont"):
        try:
            tkfont.nametofont(name).configure(family=family)
        except tk.TclError:
            pass


class PreviewWindow(ctk.CTkToplevel):
    """Read-only preview of a finished .srt file, with copy-to-clipboard.

    The text is selectable (select + Ctrl+C works on a disabled CTkTextbox);
    "kopii ĉion" copies the whole file to the clipboard.
    """

    def __init__(self, master: tk.Misc, srt_path: Path):
        super().__init__(master)
        self.title(f"antaŭrigardo: {srt_path.name}")
        self.geometry("720x480")
        self.minsize(480, 320)
        try:
            self._text = srt_path.read_text(encoding="utf-8")
        except OSError as exc:
            self._text = f"(ne eblas legi {srt_path.name}: {exc})"
        self._box = ctk.CTkTextbox(self, wrap="word", corner_radius=0)
        self._box.pack(fill="both", expand=True, padx=10, pady=(10, 6))
        self._box.insert("1.0", self._text)
        self._box.configure(state="disabled")  # read-only, but still selectable
        btns = ctk.CTkFrame(self, fg_color="transparent", corner_radius=0)
        btns.pack(fill="x", padx=10, pady=(0, 10))
        self._copy_btn = ctk.CTkButton(btns, text="kopii ĉion", command=self._copy_all, corner_radius=0)
        self._copy_btn.pack(side="left")
        ctk.CTkButton(btns, text="fermi", command=self.destroy, corner_radius=0).pack(side="right")

    def _copy_all(self) -> None:
        self.clipboard_clear()
        self.clipboard_append(self._text)
        self._copy_btn.configure(text="kopiita!")
        self.after(1500, self._revert_copy_btn)

    def _revert_copy_btn(self) -> None:
        if self.winfo_exists():
            self._copy_btn.configure(text="kopii ĉion")


class App(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("eo-stt - Esperanto parolrekonado (mp3 -> srt)")
        self.geometry("760x560")
        self.minsize(680, 460)
        family = _default_font_family(self)
        ctk.ThemeManager.theme["CTkFont"]["family"] = family
        _setup_tk_fonts(self, family)
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
        self._placeholder = ctk.CTkLabel(self._list_frame, text="(ankorau neniuj dosieroj)", anchor="w")
        self._placeholder.pack(fill="x", pady=2)
        btns = ctk.CTkFrame(top, fg_color="transparent", **cr)
        btns.pack(side="left", fill="y", padx=(8, 0))
        ctk.CTkButton(btns, text="aldoni dosierojn...", command=self._add_files, **cr).pack(fill="x", pady=2)
        ctk.CTkButton(btns, text="malplenigi", command=self._clear_files, **cr).pack(fill="x", pady=2)

        out = ctk.CTkFrame(self, fg_color="transparent", **cr)
        out.pack(fill="x", **pad)
        ctk.CTkLabel(out, text="eliga dosierujo:").pack(side="left")
        self._out_var = tk.StringVar()
        ctk.CTkEntry(out, textvariable=self._out_var, **cr).pack(side="left", fill="x", expand=True, padx=6)
        ctk.CTkButton(out, text="elekti...", command=self._browse_out, **cr).pack(side="left")

        run = ctk.CTkFrame(self, fg_color="transparent", **cr)
        run.pack(fill="x", **pad)
        self._run_btn = ctk.CTkButton(run, text="transskribu -> .srt", command=self._start, **cr)
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
        for path in _ask_open_filenames("elektu aŭdiajn dosierojn"):
            if path in self._rows:
                continue
            row = ctk.CTkFrame(self._list_frame, fg_color="transparent", corner_radius=0)
            row.pack(fill="x", pady=1)
            ctk.CTkLabel(row, text=path.name, anchor="w").pack(side="left", fill="x", expand=True)
            ctk.CTkButton(
                row, text="srt", width=36, height=28, corner_radius=0,
                command=lambda p=path: self._preview_file(p),
            ).pack(side="right", padx=2)
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
        directory = _ask_directory("elektu eligan dosierujon")
        if directory:
            self._out_var.set(directory)

    # -- preview ------------------------------------------------------------

    def _preview_file(self, audio_path: Path) -> None:
        """Open the preview window for the .srt that belongs to audio_path."""
        out_dir = self._out_var.get().strip() or None
        self._open_preview(srt_path_for(audio_path, out_dir))

    def _open_preview(self, srt_path: Path) -> None:
        if not srt_path.is_file():
            messagebox.showinfo("eo-stt", "ĉi tiu dosiero ankoraŭ ne estas transskribita.")
            return
        PreviewWindow(self, srt_path)

    # -- transcription ----------------------------------------------------

    def _start(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        if not self._files:
            messagebox.showinfo("eo-stt", "unue elektu almenaŭ unu aŭdan dosieron.")
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
            self._events.put(("log", "preparante modelon (INT8 ONNX, CPU)..."))
            model = load_model()
            last_out: Path | None = None
            for i, f in enumerate(files, 1):
                self._events.put(("progress", (i - 1) / n_files * 100))
                self._events.put(("log", f"[{i}/{n_files}] {f.name} ..."))
                base = (i - 1) / n_files * 100
                step = 100 / n_files

                def on_progress(done: int, total: int, base: float = base, step: float = step) -> None:
                    self._events.put(("progress", base + step * done / total))

                out = transcribe_file(model, f, out_dir, on_progress=on_progress)
                last_out = out
                self._events.put(("log", f"    skribita: {out}"))
            self._events.put(("progress", 100))
            self._events.put(("log", "farita."))
            self._events.put(("done", str(last_out) if last_out is not None else None))
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
                    # all files transcribed: open a preview of the last .srt
                    if payload:
                        self._open_preview(Path(payload))
                elif kind == "error":
                    messagebox.showerror("eo-stt - eraro", payload)
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
