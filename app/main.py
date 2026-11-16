"""Application entry point, PyInstaller asset path resolution, and QApplication bootstrap."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Sequence

from PySide6.QtWidgets import QApplication

# Ensure project root is importable when running `python app/main.py` directly
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if not getattr(sys, "frozen", False) and str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.ui.main_window import MainWindow


def resolve_asset_path(relative: str) -> Path:
    """Resolve an asset/config path relative to PyInstaller _MEIPASS or the workspace root."""
    rel = Path(relative)
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        meipass = Path(sys._MEIPASS)  # type: ignore[attr-defined]
        candidate = meipass / rel
        if candidate.exists():
            return candidate
        exe_candidate = Path(sys.executable).resolve().parent / rel
        if exe_candidate.exists():
            return exe_candidate
        return candidate
    return _PROJECT_ROOT / rel


def load_stylesheet(app: QApplication, relative_qss: str = "app/ui/theme.qss") -> bool:
    """Load and apply the dark QSS theme onto QApplication with resolved icon URLs."""
    import re

    qss_path = resolve_asset_path(relative_qss)
    if qss_path.is_file():
        qss_text = qss_path.read_text(encoding="utf-8")

        def _rewrite_url(match: re.Match[str]) -> str:
            rel = match.group(1).strip().strip("\"'")
            resolved = resolve_asset_path(rel).as_posix()
            return f"url({resolved})"

        qss_text = re.sub(r"url\(([^)]+)\)", _rewrite_url, qss_text)
        app.setStyleSheet(qss_text)
        return True
    return False


def create_application(argv: Optional[Sequence[str]] = None) -> tuple[QApplication, MainWindow]:
    """Create or reuse QApplication, load theme.qss, and build MainWindow with an initial tab."""
    existing = QApplication.instance()
    if isinstance(existing, QApplication):
        app = existing
    else:
        args = list(argv) if argv is not None else sys.argv
        app = QApplication(args)

    app.setApplicationName("LitePhotoEditor")
    app.setOrganizationName("LitePhotoEditor")
    load_stylesheet(app, "app/ui/theme.qss")

    window = MainWindow(create_initial_doc=True)
    return app, window


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Launch the LitePhotoEditor GUI event loop."""
    app, window = create_application(argv=argv)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
