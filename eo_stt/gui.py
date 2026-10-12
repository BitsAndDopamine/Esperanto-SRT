"""Tkinter GUI (Linux + Windows, stdlib only).

Run with: python -m eo_stt

All transcription happens in a worker thread; the GUI only receives events
through a queue (Tk is not thread-safe).
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from .model import MODEL_ID, load_model
from .transcribe import transcribe_file

AUDIO_TYPES = [("Audio files", "*.mp3 *.wav *.ogg *.flac *.m4a"), ("All files", "*.*")]


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("eo-stt — Esperanto speech-to-text (mp3 → srt)")
        self.minsize(640, 420)
        self._events: queue.Queue = queue.Queue()
        self._worker: threading.Thread | None = None
        self._files: list[Path] = []
        self._build()
        self.after(100, self._poll)

    def _build(self) -> None:
        pad = {"padx": 8, "pady": 4}

        top = ttk.Frame(self)
        top.pack(fill="x", **pad)
        self._listbox = tk.Listbox(top, selectmode="extended", height=6)
        self._listbox.pack(side="left", fill="both", expand=True)
        btns = ttk.Frame(top)
        btns.pack(side="left", fill="y", padx=(8, 0))
        ttk.Button(btns, text="Add files…", command=self._add_files).pack(fill="x", pady=2)
        ttk.Button(btns, text="Remove selected", command=self._remove_selected).pack(fill="x", pady=2)
        ttk.Button(btns, text="Clear", command=self._clear_files).pack(fill="x", pady=2)

        out = ttk.Frame(self)
        out.pack(fill="x", **pad)
        ttk.Label(out, text="Output folder:").pack(side="left")
        self._out_var = tk.StringVar()
        ttk.Entry(out, textvariable=self._out_var).pack(side="left", fill="x", expand=True, padx=4)
        ttk.Button(out, text="Browse…", command=self._browse_out).pack(side="left")

        run = ttk.Frame(self)
        run.pack(fill="x", **pad)
        self._run_btn = ttk.Button(run, text="Transcribe → .srt", command=self._start)
        self._run_btn.pack(side="left")
        self._progress = ttk.Progressbar(run, mode="determinate", maximum=100)
        self._progress.pack(side="left", fill="x", expand=True, padx=8)

        self._log = ScrolledText(self, height=12, state="disabled")
        self._log.pack(fill="both", expand=True, **pad)

    # -- widget helpers ---------------------------------------------------

    def _append_log(self, msg: str) -> None:
        self._log.configure(state="normal")
        self._log.insert("end", msg + "\n")
        self._log.see("end")
        self._log.configure(state="disabled")

    def _add_files(self) -> None:
        paths = filedialog.askopenfilenames(title="Choose audio files", filetypes=AUDIO_TYPES)
        for p in paths:
            path = Path(p)
            if path not in self._files:
                self._files.append(path)
                self._listbox.insert("end", path.name)
        if self._files and not self._out_var.get():
            self._out_var.set(str(self._files[0].parent))

    def _remove_selected(self) -> None:
        for i in reversed(self._listbox.curselection()):
            self._listbox.delete(i)
            del self._files[i]

    def _clear_files(self) -> None:
        self._listbox.delete(0, "end")
        self._files.clear()

    def _browse_out(self) -> None:
        directory = filedialog.askdirectory(title="Choose output folder")
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
        self._progress["value"] = 0
        self._worker = threading.Thread(target=self._work, daemon=True)
        self._worker.start()

    def _work(self) -> None:
        files = list(self._files)
        out_dir = self._out_var.get().strip() or None
        try:
            self._events.put(("log", f"Loading model {MODEL_ID} (first run downloads ~500 MB)…"))
            model = load_model()
            for i, f in enumerate(files, 1):
                self._events.put(("progress", (i - 1) / len(files) * 100))
                self._events.put(("log", f"[{i}/{len(files)}] {f.name} …"))
                out = transcribe_file(model, f, out_dir)
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
                    self._progress["value"] = payload
                elif kind == "done":
                    messagebox.showinfo("eo-stt", "All files transcribed.")
                elif kind == "error":
                    messagebox.showerror("eo-stt — error", payload)
                elif kind == "finished":
                    self._run_btn.configure(state="normal")
        except queue.Empty:
            pass
        self.after(100, self._poll)


def main() -> None:
    App().mainloop()


if __name__ == "__main__":
    main()
