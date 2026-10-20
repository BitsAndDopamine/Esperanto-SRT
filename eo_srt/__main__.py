"""Entry point: `python -m eo_srt` or the PyInstaller executable.

No arguments -> GUI. With arguments -> CLI (eo-srt FILE.mp3 [...] -o OUT_DIR).
"""

import sys


def main() -> None:
    if len(sys.argv) > 1:
        from .cli import main as cli_main

        sys.exit(cli_main())
    from .gui import main as gui_main

    gui_main()


if __name__ == "__main__":
    main()
