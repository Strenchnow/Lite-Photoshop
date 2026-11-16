"""Build script for compiling core_engine.dll and packaging LitePhotoEditor via PyInstaller."""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import os
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from c_core.compile_dll import compile_core_dll, get_library_filename  # noqa: E402


APP_NAME = "LitePhotoEditor"
ENTRY_POINT = PROJECT_ROOT / "app" / "main.py"
THEME_QSS = PROJECT_ROOT / "app" / "ui" / "theme.qss"
KEYBINDINGS_JSON = PROJECT_ROOT / "app" / "config" / "keybindings.json"
ASSETS_DIR = PROJECT_ROOT / "assets"
MASTER_ICON_PNG = PROJECT_ROOT / "icon.png"
ICON_ICO = ASSETS_DIR / "icon.ico"
ICON_256_PNG = ASSETS_DIR / "icon_256.png"
ICO_SIZES: tuple[tuple[int, int], ...] = (
    (16, 16),
    (24, 24),
    (32, 32),
    (48, 48),
    (64, 64),
    (128, 128),
    (256, 256),
)


def generate_app_icons(
    source_png: Optional[Path] = None,
    assets_dir: Optional[Path] = None,
    force: bool = False,
) -> tuple[Path, Path]:
    """Generate multi-resolution assets/icon.ico and lightweight assets/icon_256.png from icon.png."""
    from PIL import Image

    src = Path(source_png) if source_png is not None else MASTER_ICON_PNG
    out_dir = Path(assets_dir) if assets_dir is not None else ASSETS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    ico_path = out_dir / "icon.ico"
    png_256_path = out_dir / "icon_256.png"

    if not src.is_file():
        if ico_path.is_file() and png_256_path.is_file():
            return ico_path, png_256_path
        raise FileNotFoundError(f"Master icon not found: {src}")

    src_mtime = src.stat().st_mtime
    needs_ico = force or not ico_path.is_file() or ico_path.stat().st_mtime < src_mtime
    needs_256 = force or not png_256_path.is_file() or png_256_path.stat().st_mtime < src_mtime

    if not needs_ico and not needs_256:
        return ico_path, png_256_path

    with Image.open(src) as im:
        rgba = im.convert("RGBA")
        if needs_256:
            resized_256 = rgba.resize((256, 256), Image.Resampling.LANCZOS)
            resized_256.save(png_256_path, format="PNG", optimize=True)
        if needs_ico:
            rgba.save(ico_path, format="ICO", sizes=list(ICO_SIZES))

    return ico_path, png_256_path


def ensure_core_dll(force: bool = False) -> Path:
    """Compile (if missing/outdated) and verify the C core shared library loads cleanly."""
    dll_path = compile_core_dll(force=force)
    if not dll_path.is_file():
        raise FileNotFoundError(f"Compiled shared library not found: {dll_path}")
    lib = ctypes.CDLL(str(dll_path))
    if not hasattr(lib, "composite_layer") or not hasattr(lib, "connected_components_alpha"):
        raise RuntimeError(f"Shared library missing expected C symbols: {dll_path}")
    return dll_path


def get_bundle_data_entries(dll_path: Optional[Path] = None) -> list[tuple[Path, str]]:
    """Return (source_path, bundle_dest_dir) pairs passed to PyInstaller --add-data."""
    lib_path = dll_path if dll_path is not None else (PROJECT_ROOT / "c_core" / get_library_filename())
    return [
        (lib_path, "c_core"),
        (THEME_QSS, "app/ui"),
        (KEYBINDINGS_JSON, "app/config"),
        (ASSETS_DIR, "assets"),
    ]


def build_pyinstaller_command(
    dll_path: Optional[Path] = None,
    onefile: bool = False,
    dist_dir: Optional[Path] = None,
    work_dir: Optional[Path] = None,
    icon_path: Optional[Path] = None,
) -> list[str]:
    """Construct the PyInstaller CLI command to package LitePhotoEditor."""
    ico = Path(icon_path) if icon_path is not None else ICON_ICO
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--name",
        APP_NAME,
        "--noconfirm",
        "--clean",
        "--windowed",
        "--paths",
        str(PROJECT_ROOT),
    ]
    if ico.is_file():
        cmd.extend(["--icon", str(ico)])
    if onefile:
        cmd.append("--onefile")
    if dist_dir is not None:
        cmd.extend(["--distpath", str(dist_dir)])
    if work_dir is not None:
        cmd.extend(["--workpath", str(work_dir)])

    sep = os.pathsep
    for src, dst in get_bundle_data_entries(dll_path=dll_path):
        cmd.extend(["--add-data", f"{src}{sep}{dst}"])

    cmd.append(str(ENTRY_POINT))
    return cmd


def validate_build_prerequisites(force_dll: bool = False) -> tuple[bool, list[str], Path]:
    """Validate DLL compilation, entry point, bundled assets, and PyInstaller availability."""
    errors: list[str] = []
    dll_path = PROJECT_ROOT / "c_core" / get_library_filename()

    try:
        dll_path = ensure_core_dll(force=force_dll)
    except Exception as exc:
        errors.append(f"C core DLL check failed: {exc}")

    try:
        ico_path, png_256_path = generate_app_icons()
        if not ico_path.is_file() or not png_256_path.is_file():
            errors.append("Failed to generate assets/icon.ico or assets/icon_256.png")
    except Exception as exc:
        errors.append(f"Icon generation failed: {exc}")

    if not ENTRY_POINT.is_file():
        errors.append(f"Entry point missing: {ENTRY_POINT}")
    if not THEME_QSS.is_file():
        errors.append(f"Theme stylesheet missing: {THEME_QSS}")
    if not KEYBINDINGS_JSON.is_file():
        errors.append(f"Keybindings config missing: {KEYBINDINGS_JSON}")
    if not ASSETS_DIR.is_dir():
        errors.append(f"Assets directory missing: {ASSETS_DIR}")

    if importlib.util.find_spec("PyInstaller") is None:
        errors.append("PyInstaller package is not installed in this Python environment.")

    return len(errors) == 0, errors, dll_path


def run_build(
    check_only: bool = False,
    force_dll: bool = False,
    onefile: bool = False,
    dist_dir: Optional[Path] = None,
    work_dir: Optional[Path] = None,
) -> int:
    """Validate prerequisites and optionally run PyInstaller to produce the standalone app."""
    import subprocess

    ok, errors, dll_path = validate_build_prerequisites(force_dll=force_dll)
    cmd = build_pyinstaller_command(
        dll_path=dll_path,
        onefile=onefile,
        dist_dir=dist_dir,
        work_dir=work_dir,
    )

    if not ok:
        print("Build prerequisite check FAILED:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print(f"Verified DLL: {dll_path} ({dll_path.stat().st_size} bytes)")
    print(f"Verified entry point: {ENTRY_POINT}")
    print(f"Verified assets & configs: {THEME_QSS}, {KEYBINDINGS_JSON}, {ASSETS_DIR}")
    print("PyInstaller command:", " ".join(cmd))

    if check_only:
        print("Check completed successfully (--check mode; skipped .exe generation).")
        return 0

    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=False)
    return int(proc.returncode)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build standalone LitePhotoEditor executable.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate DLL, entry point, assets, and PyInstaller without creating a full .exe.",
    )
    parser.add_argument(
        "--force-dll",
        action="store_true",
        help="Force recompilation of c_core/core_engine.dll before packaging.",
    )
    parser.add_argument(
        "--onefile",
        action="store_true",
        help="Package into a single executable file via PyInstaller --onefile.",
    )
    parser.add_argument("--distpath", type=Path, default=None, help="Optional PyInstaller dist directory.")
    parser.add_argument("--workpath", type=Path, default=None, help="Optional PyInstaller work directory.")
    args = parser.parse_args(argv)

    return run_build(
        check_only=args.check,
        force_dll=args.force_dll,
        onefile=args.onefile,
        dist_dir=args.distpath,
        work_dir=args.workpath,
    )


if __name__ == "__main__":
    raise SystemExit(main())
