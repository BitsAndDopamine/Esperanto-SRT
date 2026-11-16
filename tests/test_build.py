"""Tests for build.py's executable naming (no PyInstaller run needed).

The Linux executable name must not end in ".<digit>": the shared MIME
database maps "*.1".."*.9" to man pages (application/x-troff-man, a text
type), so a file named e.g. "eo-srt-v0.2" is opened as a text file by Linux
file managers on double-click instead of being run. build.py therefore
appends "-linux" (Windows keeps the ".exe" extension, which has no such
rule). Verified against the real MIME database with:
    xdg-mime query filetype <file>
"""

import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import build  # noqa: E402
from eo_srt import __version__  # noqa: E402


def test_linux_exe_name_has_version_but_no_trailing_digit():
    name = build.versioned_exe_name("eo-srt", "v0.2", "linux")
    assert name == "eo-srt-v0.2-linux"
    assert name.endswith("-linux")


def test_windows_exe_name():
    assert build.versioned_exe_name("eo-srt", "v0.2", "win32") == "eo-srt-v0.2.exe"


def test_darwin_exe_name():
    # macOS: "eo-srt-v0.2-macos" — the "-macos" suffix keeps the version in the
    # name without a trailing digit (same man-page glob issue as Linux) and
    # marks the platform in the artifact name (eo-srt-<tag>-macos.tar.gz).
    assert build.versioned_exe_name("eo-srt", "v0.2", "darwin") == "eo-srt-v0.2-macos"


def test_ci_workflow_builds_macos_on_hosted_runners():
    # The macOS port needs no Mac: CI builds it on GitHub's hosted macOS
    # runners (Apple Silicon arm64 + Intel x86_64), same onedir layout as Linux.
    wf = (Path(__file__).resolve().parent.parent / ".github" / "workflows" / "build.yml").read_text(
        encoding="utf-8"
    )
    assert "macos-latest" in wf       # Apple Silicon (arm64)
    assert "macos-26-intel" in wf      # Intel (x86_64)
    assert "label: macos\n" in wf
    assert "label: macos-intel" in wf
    # macOS ships a double-clickable .app bundle: smoke-test and pack it
    assert "eo-srt.app/Contents/MacOS/eo-srt-*" in wf
    assert "tar -czf \"dist/eo-srt-${{ github.ref_name }}-${{ matrix.label }}.tar.gz\" -C dist eo-srt.app" in wf


def test_exe_name_never_ends_in_digit():
    # a trailing digit would hit the "*.1".."*.9" man-page globs (Linux)
    for platform in ("linux", "win32", "darwin"):
        name = build.versioned_exe_name("eo-srt", "v0.2", platform)
        assert not name[-1].isdigit(), f"{platform}: {name} ends in a digit"


def test_version_label_defaults_to_package_version():
    assert build.versioned_exe_name("eo-srt", None, "linux") == f"eo-srt-v{__version__}-linux"
    assert build.versioned_exe_name("eo-srt", None, "win32") == f"eo-srt-v{__version__}.exe"


def test_icon_png_exists_and_is_a_valid_png():
    icon = build.ICON_PNG
    assert icon.is_file()
    data = icon.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic
    # IHDR: width/height are big-endian uint32 at offsets 16 and 20
    width, height = struct.unpack(">II", data[16:24])
    assert width >= 64 and height >= 64


def test_exe_icon_is_the_png_on_windows_only(monkeypatch):
    # PyInstaller embeds the icon into the exe on Windows (converting the PNG
    # to .ico); a Linux ELF binary carries no icon (taskbar icon comes from
    # the running window instead)
    monkeypatch.setattr(sys, "platform", "win32")
    assert build.exe_icon() == str(build.ICON_PNG)
    monkeypatch.setattr(sys, "platform", "linux")
    assert build.exe_icon() is None


def _format_spec(**overrides):
    kwargs = dict(
        collect_all=[], copy_metadata=[], exclude=[], main_py="main.py",
        root=".", name="eo-srt", exe_name="eo-srt", console=True,
        model_datas="", bundle_block="", icon=None, icon_png="eo-srt.png",
    )
    kwargs.update(overrides)
    return build.SPEC_TEMPLATE.format(**kwargs)


def test_spec_template_embeds_and_bundles_the_icon():
    spec = _format_spec(icon="eo-srt.png")
    assert "icon='eo-srt.png'" in spec      # embedded into the Windows exe
    assert "('eo-srt.png', '.')" in spec    # bundled for the window/taskbar icon


def test_linux_windows_spec_has_no_app_bundle_and_keeps_console():
    spec = _format_spec()  # linux/windows defaults: plain onedir, console=True
    assert "BUNDLE(" not in spec
    assert "console=True" in spec
    assert "name='eo-srt'" in spec


def test_macos_spec_builds_app_bundle_with_versioned_exe_and_model_inside():
    bundle_block = build.BUNDLE_BLOCK.format(
        app_name="eo-srt.app", app_icon="eo-srt.png",
        bundle_identifier=build.BUNDLE_IDENTIFIER, version="1.0",
    )
    spec = _format_spec(
        exe_name="eo-srt-v1.0-macos",
        console=False,  # windowed .app: Dock icon + menu bar on double-click
        model_datas="datas.append(('models/x', 'model'))",
        bundle_block=bundle_block,
    )
    assert "BUNDLE(" in spec
    assert "name='eo-srt.app'" in spec
    # the versioned exe name must reach EXE itself, so CFBundleExecutable in
    # Info.plist matches the binary in Contents/MacOS (no post-build rename)
    assert "name='eo-srt-v1.0-macos'" in spec
    assert "console=False" in spec
    # the model ships as data inside the .app (Contents/Resources, cross-linked
    # to Contents/Frameworks = _MEIPASS), not beside the executable
    assert "datas.append(('models/x', 'model'))" in spec
    assert f"bundle_identifier='{build.BUNDLE_IDENTIFIER}'" in spec
