"""PyInstaller entry point for the portable executable.

Also runnable from source: `python main.py` (no args -> GUI, args -> CLI).
Kept at the repo root (outside the package) so PyInstaller's static analysis
resolves the eo_stt package cleanly.
"""

import sys

if len(sys.argv) > 1:
    from eo_stt.cli import main as cli_main

    sys.exit(cli_main())

from eo_stt.gui import main as gui_main

gui_main()
