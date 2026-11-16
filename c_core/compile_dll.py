"""Cross-compiler detection and build helper for c_core/core_engine.dll."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


CORE_DIR = Path(__file__).resolve().parent
SRC_FILE = CORE_DIR / "core_engine.c"
HDR_FILE = CORE_DIR / "core_engine.h"


def get_library_filename() -> str:
    """Return the platform-appropriate shared library filename."""
    if sys.platform == "win32":
        return "core_engine.dll"
    if sys.platform == "darwin":
        return "libcore_engine.dylib"
    return "libcore_engine.so"


def _needs_rebuild(src: Path, hdr: Path, target: Path) -> bool:
    if not target.exists():
        return True
    target_mtime = target.stat().st_mtime
    if src.stat().st_mtime > target_mtime:
        return True
    if hdr.exists() and hdr.stat().st_mtime > target_mtime:
        return True
    return False


def _run_cmd(cmd: list[str], cwd: Path) -> tuple[bool, str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return proc.returncode == 0, proc.stdout.strip()


def _verify_loadable(target: Path) -> bool:
    """Verify the built DLL can be loaded by Python without extra PATH dirs."""
    import ctypes

    try:
        lib = ctypes.CDLL(str(target))
        _ = lib.composite_layer
        return True
    except OSError:
        return False


def _try_gcc(compiler: str, src: Path, target: Path) -> tuple[bool, str]:
    common = [compiler, "-O3", "-shared", "-std=c11", "-Wall", "-Wextra", "-Wno-unused-parameter"]
    if sys.platform != "win32":
        common.append("-fPIC")

    candidates: list[list[str]] = []
    if sys.platform == "win32":
        # Prefer fully self-contained DLL with static OpenMP + static libgcc/winpthread
        candidates.append(
            common
            + [
                "-fopenmp",
                "-static-libgcc",
                "-Wl,-Bstatic",
                "-lgomp",
                "-lwinpthread",
                "-Wl,-Bdynamic",
                "-o",
                str(target),
                str(src),
                "-lm",
            ]
        )
        # Fallback: single-threaded self-contained DLL (zero external runtime DLLs)
        candidates.append(
            common
            + [
                "-static-libgcc",
                "-o",
                str(target),
                str(src),
                "-lm",
            ]
        )
    else:
        candidates.append(common + ["-fopenmp", "-o", str(target), str(src), "-lm"])
        candidates.append(common + ["-o", str(target), str(src), "-lm"])

    last_out = ""
    for cmd in candidates:
        ok, out = _run_cmd(cmd, CORE_DIR)
        last_out = out
        if ok and target.exists() and _verify_loadable(target):
            return True, out
    return False, last_out


def _try_clang(compiler: str, src: Path, target: Path) -> tuple[bool, str]:
    common = [compiler, "-O3", "-shared", "-std=c11", "-Wall", "-Wextra"]
    if sys.platform != "win32":
        common.append("-fPIC")

    candidates = [
        common + ["-fopenmp", "-o", str(target), str(src), "-lm"],
        common + ["-o", str(target), str(src), "-lm"],
    ]
    if sys.platform == "win32":
        candidates = [
            common + ["-o", str(target), str(src)],
        ]

    last_out = ""
    for cmd in candidates:
        ok, out = _run_cmd(cmd, CORE_DIR)
        last_out = out
        if ok and target.exists() and _verify_loadable(target):
            return True, out
    return False, last_out


def _try_msvc(compiler: str, src: Path, target: Path) -> tuple[bool, str]:
    cmd = [
        compiler,
        "/nologo",
        "/O2",
        "/LD",
        "/openmp",
        str(src),
        f"/Fe:{target}",
    ]
    ok, out = _run_cmd(cmd, CORE_DIR)
    if ok and target.exists() and _verify_loadable(target):
        return True, out

    cmd_no_omp = [
        compiler,
        "/nologo",
        "/O2",
        "/LD",
        str(src),
        f"/Fe:{target}",
    ]
    ok2, out2 = _run_cmd(cmd_no_omp, CORE_DIR)
    if ok2 and target.exists() and _verify_loadable(target):
        return True, out2
    return False, out2 or out


def _find_compiler(preferred: Optional[str] = None) -> list[tuple[str, str]]:
    """Return ordered list of (kind, executable_path) candidates found on system."""
    found: list[tuple[str, str]] = []
    search_order = ["gcc", "clang", "cl"]
    if preferred and preferred in search_order:
        search_order.remove(preferred)
        search_order.insert(0, preferred)

    for name in search_order:
        exe = shutil.which(name)
        if exe:
            found.append((name, exe))

    # Common MSYS2 / MinGW locations on Windows if not on PATH
    if sys.platform == "win32" and not any(k == "gcc" for k, _ in found):
        for candidate in (
            Path(r"C:\msys64\ucrt64\bin\gcc.exe"),
            Path(r"C:\msys64\mingw64\bin\gcc.exe"),
            Path(r"C:\mingw64\bin\gcc.exe"),
        ):
            if candidate.exists():
                found.append(("gcc", str(candidate)))
                break

    return found


def compile_core_dll(force: bool = False, preferred_compiler: Optional[str] = None) -> Path:
    """Compile core_engine.c into core_engine.dll if missing, outdated, or forced."""
    if not SRC_FILE.exists():
        raise FileNotFoundError(f"C source file not found: {SRC_FILE}")

    target = CORE_DIR / get_library_filename()
    if not force and not _needs_rebuild(SRC_FILE, HDR_FILE, target):
        return target

    compilers = _find_compiler(preferred=preferred_compiler)
    if not compilers:
        raise RuntimeError(
            "No supported C compiler found (tried gcc, clang, cl.exe). "
            "Install MSYS2/MinGW gcc, Clang, or MSVC Build Tools."
        )

    errors: list[str] = []
    for kind, exe in compilers:
        if kind == "gcc":
            ok, out = _try_gcc(exe, SRC_FILE, target)
        elif kind == "clang":
            ok, out = _try_clang(exe, SRC_FILE, target)
        else:
            ok, out = _try_msvc(exe, SRC_FILE, target)

        if ok:
            # Clean up MSVC intermediate files if produced
            for ext in (".obj", ".exp", ".lib"):
                leftover = CORE_DIR / f"core_engine{ext}"
                if leftover.exists():
                    try:
                        leftover.unlink()
                    except OSError:
                        pass
            return target

        errors.append(f"[{kind}: {exe}]\n{out}")

    raise RuntimeError(
        "Failed to compile core_engine shared library with all available compilers:\n"
        + "\n---\n".join(errors)
    )


def main() -> int:
    force = "--force" in sys.argv or "-f" in sys.argv
    try:
        out_path = compile_core_dll(force=force)
        print(f"Compiled successfully: {out_path} ({out_path.stat().st_size} bytes)")
        return 0
    except Exception as exc:
        print(f"Compilation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
