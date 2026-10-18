"""The version lives in exactly one place: eo_stt.__version__.

pyproject.toml reads it dynamically (setuptools attr), the GUI's "pri"
window shows it, and build.py embeds it in the executable name. This test
keeps the installed package metadata from drifting away from the code.
"""

import importlib.metadata

from eo_stt import __version__


def test_version_is_dotted_numbers():
    parts = __version__.split(".")
    assert len(parts) >= 2
    assert all(part.isdigit() for part in parts)


def test_installed_package_version_matches():
    assert importlib.metadata.version("eo-stt") == __version__
