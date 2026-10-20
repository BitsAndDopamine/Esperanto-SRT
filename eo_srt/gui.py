"""CustomTkinter GUI (Linux + Windows + macOS, dark mode).

Run with: python -m eo_srt   (or the PyInstaller executable with no arguments)

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

from . import __version__, subprocess_env
from .model import load_model
from .transcribe import srt_path_for, transcribe_file

# File-type dropdown of the open dialog. The first entry (all audio formats,
# preselected) is kept, and each supported format is listed separately so the
# dropdown actually filters: some native dialogs mishandle one entry with
# several patterns, and per-format entries let the user pick a single format.
AUDIO_TYPES = [
    ("aŭdiaj dosieroj", "*.mp3 *.wav *.ogg *.flac"),
    ("mp3-dosieroj (*.mp3)", "*.mp3"),
    ("wav-dosieroj (*.wav)", "*.wav"),
    ("ogg-dosieroj (*.ogg)", "*.ogg"),
    ("flac-dosieroj (*.flac)", "*.flac"),
    ("ĉiuj dosieroj", "*"),
]

# Share of the progress bar reserved for loading the model. Loading the INT8
# ONNX model takes a while on CPU, so it gets its own visible first step.
MODEL_LOAD_SHARE = 0.05

# App icon: <repo>/eo-srt.png when running from source, the PyInstaller bundle
# (_MEIPASS) when frozen. Used for the window/taskbar icon (App._set_icon).
ICON_FILE = "eo-srt.png"


def _icon_path() -> Path:
    """Locate the app icon: repo root from source, the bundle dir when frozen."""
    if getattr(sys, "frozen", False):
        meipass = Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
        return meipass / ICON_FILE
    return Path(__file__).resolve().parent.parent / ICON_FILE

# Text of the "pri" (about) window: what the app is, where the source lives,
# and which licenses apply (the code and the bundled model differ).
ABOUT_TEXT = (
    f"esperanto-srt v{__version__}\n"
    "esperanta parolrekonado (mp3 -> srt)\n"
    "\n"
    "github: https://github.com/BitsAndDopamine/Esperanto-SRT\n"
    "(kodo, novaj versioj kaj por viaj raportoj pri cimnoj)\n"
    "\n"
    "licenco de la kodo: CC0 1.0 Universal\n"
    "licenco de la modelo: CC-BY-4.0\n"
    "nvidia-eo-conformer-transducer-large-onnx\n"
    "\n"
    "Dankon al ciuj kontribuantoj de la projekto Common Voice!\n"
    "https://commonvoice.mozilla.org/eo \n"
    "\n"
)


def _zenity_path() -> str | None:
    """Path to zenity (the native GNOME file selector) on Linux, else None."""
    if sys.platform.startswith("linux"):
        return shutil.which("zenity")
    return None


def _run_zenity(zenity: str, args: list[str]) -> subprocess.CompletedProcess | None:
    """Run zenity with a clean environment (see eo_srt.subprocess_env).

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
        logging.getLogger("eo_srt").warning("zenity ne povis starti: %s", exc)
        return None


def _ask_open_filenames(title: str) -> list[Path]:
    """Pick audio files with the OS-native selector.

    Linux/GNOME: zenity (the native GTK file dialog) when installed.
    Windows/macOS: tkinter's filedialog (already the native OS dialog).
    Linux without zenity (or when zenity fails): tkinter's Tk dialog.
    """
    zenity = _zenity_path()
    if zenity:
        filters = [
            f"--file-filter={label} | {patterns}"
            for label, patterns in AUDIO_TYPES
            if patterns != "*"  # GTK already offers its own "all files" filter
        ]
        proc = _run_zenity(
            zenity,
            [
                "--file-selection", "--multiple",
                "--separator", "\n",
                f"--title={title}",
                *filters,
            ],
        )
        if proc is not None:
            if proc.returncode == 0:
                return [Path(p) for p in proc.stdout.splitlines() if p.strip()]
            if proc.returncode == 1:
                return []  # user cancelled
            # any other exit code is a failure, not a cancel: say so and
            # fall back to the Tk dialog instead of silently doing nothing
            logging.getLogger("eo_srt").warning(
                "zenity malsukcesis (elirkodo %s): %s",
                proc.returncode, proc.stderr.strip() or "neniu detalo",
            )
    try:
        paths = filedialog.askopenfilenames(title=title, filetypes=AUDIO_TYPES)
    except tk.TclError as exc:
        messagebox.showerror("eo-srt - eraro", f"ne eblas malfermi la dosier-elektilon:\n{exc}")
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
            logging.getLogger("eo_srt").warning(
                "zenity malsukcesis (elirkodo %s): %s",
                proc.returncode, proc.stderr.strip() or "neniu detalo",
            )
    try:
        return filedialog.askdirectory(title=title)
    except tk.TclError as exc:
        messagebox.showerror("eo-srt - eraro", f"ne eblas malfermi la dosieruj-elektilon:\n{exc}")
        return ""


class _QueueHandler(logging.Handler):
    """Forward eo_srt log records into the GUI event queue (worker thread safe)."""

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


class AboutWindow(ctk.CTkToplevel):
    """The "pri" window: version, GitHub repo and the licenses.

    Read-only like the preview, but the text stays selectable so the
    repository URL can be copied out of it.
    """

    def __init__(self, master: tk.Misc):
        super().__init__(master)
        self.title("pri eo-srt")
        self.geometry("560x300")
        self.minsize(420, 240)
        box = ctk.CTkTextbox(self, wrap="word", corner_radius=0)
        box.pack(fill="both", expand=True, padx=10, pady=(10, 6))
        box.insert("1.0", ABOUT_TEXT)
        box.configure(state="disabled")  # read-only, but still selectable
        ctk.CTkButton(self, text="fermi", command=self.destroy, corner_radius=0).pack(pady=(0, 10))


class App(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("esperanto-srt - Esperanto parolrekonado (mp3 -> srt)")
        self.geometry("760x560")
        self.minsize(680, 460)
        self._set_icon()
        family = _default_font_family(self)
        ctk.ThemeManager.theme["CTkFont"]["family"] = family
        _setup_tk_fonts(self, family)
        self._events: queue.Queue = queue.Queue()
        self._worker: threading.Thread | None = None
        self._files: list[Path] = []
        self._rows: dict[Path, ctk.CTkFrame] = {}
        self._srt_buttons: dict[Path, ctk.CTkButton] = {}
        logger = logging.getLogger("eo_srt")
        logger.setLevel(logging.INFO)
        logger.addHandler(_QueueHandler(self._events))
        self._build()
        self.after(100, self._poll)

    def _set_icon(self) -> None:
        """Window + taskbar icon (WM_SETICON on Windows, _NET_WM_ICON on
        Linux). The PhotoImage must stay referenced or Tk drops the icon."""
        try:
            icon = _icon_path()
            if icon.is_file():
                self._icon_image = tk.PhotoImage(file=str(icon))
                # -default: also applies to toplevels opened later (pri/preview)
                self.iconphoto(True, self._icon_image)
        except tk.TclError:
            pass  # a missing or unreadable icon must not keep the GUI from starting

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
        ctk.CTkButton(btns, text="pri", command=self._show_about, **cr).pack(fill="x", pady=2)

        out = ctk.CTkFrame(self, fg_color="transparent", **cr)
        out.pack(fill="x", **pad)
        ctk.CTkLabel(out, text="eliga dosierujo:").pack(side="left")
        self._out_var = tk.StringVar()
        ctk.CTkEntry(out, textvariable=self._out_var, **cr).pack(side="left", fill="x", expand=True, padx=6)
        ctk.CTkButton(out, text="elekti...", command=self._browse_out, **cr).pack(side="left")

        run = ctk.CTkFrame(self, fg_color="transparent", **cr)
        run.pack(fill="x", **pad)
        bar_row = ctk.CTkFrame(run, fg_color="transparent", **cr)
        bar_row.pack(fill="x")
        self._run_btn = ctk.CTkButton(bar_row, text="transskribu -> .srt", command=self._start, **cr)
        self._run_btn.pack(side="left")
        self._progress = ctk.CTkProgressBar(bar_row, **cr)
        self._progress.pack(side="left", fill="x", expand=True, padx=10)
        self._progress.set(0)
        # one-line status under the bar: what the app is doing right now
        self._status = ctk.CTkLabel(run, text="", anchor="w", **cr)
        self._status.pack(fill="x", pady=(2, 0))

        self._log = ctk.CTkTextbox(self, state="disabled", **cr)
        self._log.pack(fill="both", expand=True, **pad)

    # -- widget helpers ---------------------------------------------------

    def _append_log(self, msg: str) -> None:
        self._log.configure(state="normal")
        self._log.insert("end", msg + "\n")
        self._log.see("end")
        self._log.configure(state="disabled")

    def _set_status(self, text: str) -> None:
        self._status.configure(text=text)

    def _add_files(self) -> None:
        for path in _ask_open_filenames("elektu aŭdiajn dosierojn"):
            if path in self._rows:
                continue
            row = ctk.CTkFrame(self._list_frame, fg_color="transparent", corner_radius=0)
            row.pack(fill="x", pady=1)
            ctk.CTkLabel(row, text=path.name, anchor="w").pack(side="left", fill="x", expand=True)
            srt_btn = ctk.CTkButton(
                row, text="srt", width=36, height=28, corner_radius=0,
                command=lambda p=path: self._preview_file(p),
            )
            srt_btn.pack(side="right", padx=2)
            ctk.CTkButton(
                row, text="x", width=28, height=28, corner_radius=0,
                command=lambda p=path: self._remove_file(p),
            ).pack(side="right", padx=2)
            self._rows[path] = row
            self._srt_buttons[path] = srt_btn
            self._files.append(path)
        if self._files:
            self._placeholder.pack_forget()
        if self._files and not self._out_var.get():
            self._out_var.set(str(self._files[0].parent))
        # a file may already have a .srt (e.g. re-adding it after a run)
        self._refresh_srt_buttons()

    def _remove_file(self, path: Path) -> None:
        row = self._rows.pop(path, None)
        if row is not None:
            row.destroy()
        self._srt_buttons.pop(path, None)
        if path in self._files:
            self._files.remove(path)
        if not self._files:
            self._placeholder.pack(fill="x", pady=2)

    def _clear_files(self) -> None:
        for row in self._rows.values():
            row.destroy()
        self._rows.clear()
        self._srt_buttons.clear()
        self._files.clear()
        self._placeholder.pack(fill="x", pady=2)

    def _browse_out(self) -> None:
        directory = _ask_directory("elektu eligan dosierujon")
        if directory:
            self._out_var.set(directory)
            # the .srt location depends on the output folder: re-check all rows
            self._refresh_srt_buttons()

    # -- about ----------------------------------------------------------------

    def _show_about(self) -> None:
        AboutWindow(self)

    # -- srt button state -------------------------------------------------------

    def _srt_exists(self, audio_path: Path) -> bool:
        """Does the .srt for audio_path exist, given the current output folder?"""
        out_dir = self._out_var.get().strip() or None
        return srt_path_for(audio_path, out_dir).is_file()

    def _refresh_srt_buttons(self) -> None:
        """A row's "srt" button is clickable only when its .srt file exists.

        Called when files are added, when the output folder changes, and as
        worker events arrive (the .srt is written incrementally, so it appears
        while a file is being transcribed).
        """
        for path, btn in self._srt_buttons.items():
            btn.configure(state="normal" if self._srt_exists(path) else "disabled")

    # -- preview ------------------------------------------------------------

    def _preview_file(self, audio_path: Path) -> None:
        """Open the preview window for the .srt that belongs to audio_path."""
        out_dir = self._out_var.get().strip() or None
        self._open_preview(srt_path_for(audio_path, out_dir))

    def _open_preview(self, srt_path: Path) -> None:
        if not srt_path.is_file():
            messagebox.showinfo("eo-srt", "ĉi tiu dosiero ankoraŭ ne estas transskribita.")
            return
        PreviewWindow(self, srt_path)

    # -- transcription ----------------------------------------------------

    def _start(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        if not self._files:
            messagebox.showinfo("eo-srt", "unue elektu almenaŭ unu aŭdan dosieron.")
            return
        self._run_btn.configure(state="disabled")
        self._progress.set(0)
        # Loading the model takes a while on CPU and the worker thread cannot
        # report anything before it is done, so show activity right away:
        # a busy animation on the bar plus a status line.
        self._progress.start()
        self._set_status("ŝargante modelon...")
        self._append_log("preparante modelon (INT8 ONNX, CPU)...")
        self._worker = threading.Thread(target=self._work, daemon=True)
        self._worker.start()

    def _work(self) -> None:
        files = list(self._files)
        out_dir = self._out_var.get().strip() or None
        n_files = len(files)
        try:
            model = load_model()
            # the model is loaded: switch from the busy animation to real
            # progress; the files share the remaining 95% of the bar
            self._events.put(("model_loaded", MODEL_LOAD_SHARE * 100))
            last_out: Path | None = None
            file_base = MODEL_LOAD_SHARE * 100
            file_span = 100 - file_base
            for i, f in enumerate(files, 1):
                base = file_base + (i - 1) / n_files * file_span
                step = file_span / n_files
                self._events.put(("status", f"transskribas [{i}/{n_files}] {f.name}"))
                self._events.put(("progress", base))
                self._events.put(("log", f"[{i}/{n_files}] {f.name} ..."))

                def on_progress(done: int, total: int, base: float = base, step: float = step) -> None:
                    self._events.put(("progress", base + step * done / total))

                out = transcribe_file(model, f, out_dir, on_progress=on_progress)
                last_out = out
                self._events.put(("log", f"    skribita: {out}"))
            self._events.put(("progress", 100))
            self._events.put(("status", "farita."))
            self._events.put(("log", "farita."))
            self._events.put(("done", str(last_out) if last_out is not None else None))
        except Exception as exc:  # show any failure in the GUI
            self._events.put(("status", "eraro"))
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
                    self._refresh_srt_buttons()  # .srt files appear during a run
                elif kind == "status":
                    self._set_status(payload)
                    self._refresh_srt_buttons()
                elif kind == "model_loaded":
                    # model loading finished: busy animation -> real progress
                    self._progress.stop()
                    self._progress.set(payload / 100)
                elif kind == "done":
                    # all files transcribed: open a preview of the last .srt
                    if payload:
                        self._open_preview(Path(payload))
                elif kind == "error":
                    messagebox.showerror("eo-srt - eraro", payload)
                elif kind == "finished":
                    self._progress.stop()  # never leave the busy animation running
                    self._run_btn.configure(state="normal")
                    self._refresh_srt_buttons()
        except queue.Empty:
            pass
        self.after(100, self._poll)


def main() -> None:
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    App().mainloop()


if __name__ == "__main__":
    main()
