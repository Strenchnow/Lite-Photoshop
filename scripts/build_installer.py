"""Automated Windows installer pipeline: icons -> PyInstaller -> Inno Setup (ISCC.exe)."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import build_exe  # noqa: E402


INSTALLER_SCRIPT = PROJECT_ROOT / "installer" / "setup.iss"
REQUIRED_ISS_SECTIONS: tuple[str, ...] = (
    "[Setup]",
    "[Languages]",
    "[Tasks]",
    "[Files]",
    "[Icons]",
    "[Registry]",
    "[Run]",
)


def find_inno_setup_compiler() -> Optional[Path]:
    """Locate ISCC.exe on PATH or in standard Windows Inno Setup 6 directories."""
    on_path = shutil.which("ISCC") or shutil.which("ISCC.exe")
    if on_path:
        p = Path(on_path)
        if p.is_file():
            return p

    candidates = [
        Path(r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe"),
        Path(r"C:\Program Files\Inno Setup 6\ISCC.exe"),
    ]
    local_app = os.environ.get("LOCALAPPDATA")
    if local_app:
        candidates.append(Path(local_app) / "Programs" / "Inno Setup 6" / "ISCC.exe")

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def validate_iss_script(iss_path: Path = INSTALLER_SCRIPT) -> tuple[bool, list[str]]:
    """Validate that installer/setup.iss exists and contains all required sections."""
    errors: list[str] = []
    if not iss_path.is_file():
        return False, [f"Inno Setup script missing: {iss_path}"]

    text = iss_path.read_text(encoding="utf-8")
    for sec in REQUIRED_ISS_SECTIONS:
        if sec not in text:
            errors.append(f"Missing section {sec} in {iss_path}")
    if "icon.ico" not in text:
        errors.append(f"SetupIconFile icon.ico reference missing in {iss_path}")
    return len(errors) == 0, errors


def build_installer(check_only: bool = False, force_icons: bool = False) -> int:
    """Run icon generation, PyInstaller build, and Inno Setup compilation."""
    ico_path, png_256_path = build_exe.generate_app_icons(force=force_icons)
    print(f"Icons ready: {ico_path}, {png_256_path}")

    iss_ok, iss_errors = validate_iss_script(INSTALLER_SCRIPT)
    if not iss_ok:
        for err in iss_errors:
            print(f"Installer script error: {err}", file=sys.stderr)
        return 1
    print(f"Verified Inno Setup script: {INSTALLER_SCRIPT}")

    exe_rc = build_exe.run_build(check_only=check_only, onefile=False)
    if exe_rc != 0:
        return exe_rc

    iscc = find_inno_setup_compiler()
    if iscc is None:
        msg = (
            "Inno Setup Compiler (ISCC.exe) not found.\n"
            "Install Inno Setup 6 via:\n"
            "  winget install JRSoftware.InnoSetup\n"
            f"Then compile '{INSTALLER_SCRIPT}' or re-run 'python scripts/build_installer.py'."
        )
        if check_only:
            print(f"Note: {msg}")
            print("Installer check completed successfully (--check mode).")
            return 0
        print(msg, file=sys.stderr)
        return 1

    print(f"Found Inno Setup Compiler: {iscc}")
    if check_only:
        print("Installer check completed successfully (--check mode).")
        return 0

    proc = subprocess.run([str(iscc), str(INSTALLER_SCRIPT)], cwd=str(PROJECT_ROOT), check=False)
    if proc.returncode == 0:
        out_setup = PROJECT_ROOT / "dist" / "LitePhotoEditor_Setup.exe"
        print(f"Installer built successfully: {out_setup}")
    return int(proc.returncode)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build LitePhotoEditor Windows Setup Installer.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Generate icons, validate setup.iss & PyInstaller prerequisites without building .exe.",
    )
    parser.add_argument(
        "--force-icons",
        action="store_true",
        help="Force regeneration of assets/icon.ico and assets/icon_256.png from icon.png.",
    )
    args = parser.parse_args(argv)
    return build_installer(check_only=args.check, force_icons=args.force_icons)


if __name__ == "__main__":
    raise SystemExit(main())
