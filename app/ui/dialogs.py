"""Dialogs for Quick Export, Chroma Key, Grid & Smart Alpha Splitters, Batch Processing, and Hotkeys."""

from __future__ import annotations

import io
import json
import sys
import threading
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image
from PySide6.QtCore import QEvent, QObject, Qt, QThread, Signal
from PySide6.QtGui import QColor, QImage, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QKeySequenceEdit,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.config.i18n import tr
from app.core.batch import BatchOperationConfig, BatchProcessor, BatchResult
from app.core.c_bridge import ComponentBox, get_engine
from app.core.document import DocumentSession, DocumentTabManager, compute_dirty_rect
from app.core.splitter import (
    detect_alpha_sprites,
    export_slices_to_folder,
    extract_alpha_sprites,
    open_slices_as_tabs,
    split_grid,
)


DEFAULT_KEYBINDINGS: dict[str, str] = {
    "move": "V",
    "hand": "H",
    "brush": "B",
    "eraser": "E",
    "smudge": "S",
    "blur_sharpen": "R",
    "marquee": "M",
    "lasso": "L",
    "magic_wand": "W",
    "crop": "C",
    "transform": "Ctrl+T",
    "eyedropper": "I",
    "bucket_fill": "G",
    "gradient": "K",
    "text": "T",
    "new_tab": "Ctrl+N",
    "open_file": "Ctrl+O",
    "save_file": "Ctrl+S",
    "quick_export": "Ctrl+Shift+E",
    "undo": "Ctrl+Z",
    "redo": "Ctrl+Y",
    "paste_clipboard": "Ctrl+V",
    "zen_mode": "Tab",
    "center_both": "Ctrl+Alt+C",
}

ACTION_DISPLAY_NAMES: dict[str, str] = {
    "move": "Tool: Move",
    "hand": "Tool: Hand",
    "brush": "Tool: Brush",
    "eraser": "Tool: Eraser",
    "smudge": "Tool: Smudge",
    "blur_sharpen": "Tool: Blur / Sharpen",
    "marquee": "Tool: Rect Marquee",
    "lasso": "Tool: Lasso",
    "magic_wand": "Tool: Magic Wand",
    "crop": "Tool: Crop",
    "transform": "Tool: Free Transform",
    "eyedropper": "Tool: Eyedropper",
    "bucket_fill": "Tool: Bucket Fill",
    "gradient": "Tool: Linear Gradient",
    "text": "Tool: Text",
    "new_tab": "File: New Document Tab",
    "open_file": "File: Open Image",
    "save_file": "File: Save",
    "quick_export": "File: Quick Export",
    "undo": "Edit: Undo",
    "redo": "Edit: Redo",
    "paste_clipboard": "Edit: Paste / Clipboard Tab",
    "zen_mode": "View: Toggle Zen Mode",
    "center_both": "Layer: Center on Canvas",
}


def resolve_keybindings_path(path: Optional[Path] = None, for_write: bool = False) -> Path:
    """Resolve keybindings.json path in workspace or PyInstaller _MEIPASS bundle."""
    if path is not None:
        return Path(path)

    rel = Path("app") / "config" / "keybindings.json"
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        exe_path = Path(sys.executable).resolve().parent / rel
        meipass_path = Path(sys._MEIPASS) / rel  # type: ignore[attr-defined]
        if for_write:
            return exe_path if exe_path.exists() or not meipass_path.exists() else meipass_path
        if exe_path.exists():
            return exe_path
        return meipass_path

    project_root = Path(__file__).resolve().parents[2]
    return project_root / rel


def load_keybindings(path: Optional[Path] = None) -> dict[str, str]:
    """Load keybindings JSON dictionary merged over defaults."""
    bindings = dict(DEFAULT_KEYBINDINGS)
    target = resolve_keybindings_path(path=path, for_write=False)
    if target.is_file():
        with target.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        if not isinstance(raw, dict):
            raise ValueError(f"Expected JSON object in {target}, got {type(raw).__name__}")
        for key, val in raw.items():
            if isinstance(key, str) and isinstance(val, str):
                bindings[key] = val
    return bindings


def save_keybindings(bindings: dict[str, str], path: Optional[Path] = None) -> Path:
    """Persist keybindings dictionary as formatted JSON and return the file path."""
    target = resolve_keybindings_path(path=path, for_write=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    cleaned = {str(k): str(v) for k, v in bindings.items()}
    with target.open("w", encoding="utf-8") as fh:
        json.dump(cleaned, fh, indent=2)
        fh.write("\n")
    return target


def encode_image_bytes(
    rgba: np.ndarray,
    fmt: str = "PNG",
    quality: int = 90,
    scale_percent: int = 100,
) -> bytes:
    """Encode an RGBA NumPy buffer to PNG, JPEG, or WEBP bytes at the requested scale and quality."""
    if not isinstance(rgba, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(rgba).__name__}")
    if rgba.ndim != 3 or rgba.shape[2] not in (3, 4):
        raise ValueError(f"Expected image shape (H, W, 3|4), got {rgba.shape}")

    if rgba.shape[2] == 3:
        h, w = int(rgba.shape[0]), int(rgba.shape[1])
        buf = np.empty((h, w, 4), dtype=np.uint8)
        buf[:, :, :3] = rgba
        buf[:, :, 3] = 255
    else:
        buf = np.ascontiguousarray(rgba, dtype=np.uint8)

    scale = max(1, int(scale_percent))
    src_h, src_w = int(buf.shape[0]), int(buf.shape[1])
    if scale != 100:
        dst_w = max(1, int(round(src_w * scale / 100.0)))
        dst_h = max(1, int(round(src_h * scale / 100.0)))
        buf = get_engine().bilinear_resize(buf, dst_w, dst_h)

    norm_fmt = fmt.strip().upper()
    if norm_fmt == "JPG":
        norm_fmt = "JPEG"
    if norm_fmt not in ("PNG", "JPEG", "WEBP"):
        raise ValueError(f"Unsupported export format: {fmt!r}")

    q = max(1, min(100, int(quality)))
    bio = io.BytesIO()
    pil_rgba = Image.fromarray(buf, mode="RGBA")
    if norm_fmt == "JPEG":
        rgb_bg = Image.new("RGB", pil_rgba.size, (255, 255, 255))
        rgb_bg.paste(pil_rgba, mask=pil_rgba.split()[3])
        rgb_bg.save(bio, format="JPEG", quality=q, optimize=True)
    elif norm_fmt == "WEBP":
        pil_rgba.save(bio, format="WEBP", quality=q, method=4)
    else:
        pil_rgba.save(bio, format="PNG", compress_level=6)
    return bio.getvalue()


def format_byte_size(num_bytes: int) -> str:
    """Format byte count cleanly as KB or MB."""
    size = max(0, int(num_bytes))
    if size >= 1024 * 1024:
        return f"{size / (1024.0 * 1024.0):.2f} MB"
    kb = size / 1024.0
    if kb < 100.0:
        return f"{kb:.2f} KB"
    return f"{kb:.1f} KB"


def _rgba_to_qpixmap(rgba: np.ndarray, max_w: int = 320, max_h: int = 240) -> QPixmap:
    """Convert contiguous RGBA ndarray to a scaled QPixmap over checkerboard for preview."""
    h, w = int(rgba.shape[0]), int(rgba.shape[1])
    # Composite onto subtle 8px checkerboard for transparent preview clarity
    tile = 8
    yy, xx = np.indices((h, w))
    checker = ((xx // tile + yy // tile) % 2) == 0
    bg = np.empty((h, w, 3), dtype=np.float32)
    bg[checker] = (68.0, 68.0, 74.0)
    bg[~checker] = (46.0, 46.0, 52.0)

    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    fg = rgba[:, :, :3].astype(np.float32)
    comp_rgb = np.clip(np.round(fg * alpha + bg * (1.0 - alpha)), 0, 255).astype(np.uint8)
    comp_rgba = np.empty((h, w, 4), dtype=np.uint8)
    comp_rgba[:, :, :3] = comp_rgb
    comp_rgba[:, :, 3] = 255

    qimg = QImage(comp_rgba.data, w, h, w * 4, QImage.Format.Format_RGBA8888).copy()
    pix = QPixmap.fromImage(qimg)
    if w > max_w or h > max_h:
        pix = pix.scaled(max_w, max_h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
    return pix


class QuickExportDialog(QDialog):
    """Quick Export dialog with live QPixmap preview and real-time KB/MB size estimate."""

    def __init__(self, doc: DocumentSession, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.doc = doc
        self._composited = np.ascontiguousarray(doc.composite().copy(), dtype=np.uint8)
        self.last_encoded_bytes: bytes = b""
        self.exported_path: Optional[Path] = None

        self.setWindowTitle(tr("export.title"))
        self.setMinimumWidth(460)
        self._build_ui()
        self.update_preview()

    @property
    def estimated_bytes(self) -> int:
        return len(self.last_encoded_bytes)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(320, 220)
        self.preview_label.setStyleSheet("background: #232328; border: 1px solid #3c3c44; border-radius: 4px;")
        root.addWidget(self.preview_label)

        info_row = QHBoxLayout()
        self.size_label = QLabel(tr("export.est_size", size="0.00 KB"))
        self.size_label.setStyleSheet("font-weight: 600;")
        self.dimensions_label = QLabel(f"{self.doc.width} x {self.doc.height} px")
        self.dimensions_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        info_row.addWidget(self.size_label)
        info_row.addWidget(self.dimensions_label)
        root.addLayout(info_row)

        form = QFormLayout()

        self.format_combo = QComboBox()
        self.format_combo.addItems(["PNG", "JPEG", "WEBP"])
        form.addRow(tr("export.format"), self.format_combo)

        q_row = QHBoxLayout()
        self.quality_slider = QSlider(Qt.Orientation.Horizontal)
        self.quality_slider.setRange(1, 100)
        self.quality_slider.setValue(90)
        self.quality_spin = QSpinBox()
        self.quality_spin.setRange(1, 100)
        self.quality_spin.setValue(90)
        q_row.addWidget(self.quality_slider, 1)
        q_row.addWidget(self.quality_spin)
        form.addRow(tr("export.quality"), q_row)

        s_row = QHBoxLayout()
        self.scale_slider = QSlider(Qt.Orientation.Horizontal)
        self.scale_slider.setRange(25, 200)
        self.scale_slider.setValue(100)
        self.scale_spin = QSpinBox()
        self.scale_spin.setRange(25, 200)
        self.scale_spin.setSuffix("%")
        self.scale_spin.setValue(100)
        s_row.addWidget(self.scale_slider, 1)
        s_row.addWidget(self.scale_spin)
        form.addRow(tr("export.scale"), s_row)

        root.addLayout(form)

        buttons = QDialogButtonBox()
        self.export_btn = buttons.addButton(tr("export.btn"), QDialogButtonBox.ButtonRole.AcceptRole)
        self.cancel_btn = buttons.addButton(tr("dlg.cancel"), QDialogButtonBox.ButtonRole.RejectRole)
        root.addWidget(buttons)

        self.format_combo.currentTextChanged.connect(self._on_format_changed)
        self.quality_slider.valueChanged.connect(self._sync_quality_from_slider)
        self.quality_spin.valueChanged.connect(self._sync_quality_from_spin)
        self.scale_slider.valueChanged.connect(self._sync_scale_from_slider)
        self.scale_spin.valueChanged.connect(self._sync_scale_from_spin)
        self.export_btn.clicked.connect(lambda: self.export_to_file())
        self.cancel_btn.clicked.connect(self.reject)

        self._on_format_changed(self.format_combo.currentText())

    def _on_format_changed(self, fmt: str) -> None:
        is_lossy = fmt.upper() in ("JPEG", "WEBP")
        self.quality_slider.setEnabled(is_lossy)
        self.quality_spin.setEnabled(is_lossy)
        self.update_preview()

    def _sync_quality_from_slider(self, value: int) -> None:
        if self.quality_spin.value() != value:
            self.quality_spin.blockSignals(True)
            self.quality_spin.setValue(value)
            self.quality_spin.blockSignals(False)
        self.update_preview()

    def _sync_quality_from_spin(self, value: int) -> None:
        if self.quality_slider.value() != value:
            self.quality_slider.blockSignals(True)
            self.quality_slider.setValue(value)
            self.quality_slider.blockSignals(False)
        self.update_preview()

    def _sync_scale_from_slider(self, value: int) -> None:
        if self.scale_spin.value() != value:
            self.scale_spin.blockSignals(True)
            self.scale_spin.setValue(value)
            self.scale_spin.blockSignals(False)
        self.update_preview()

    def _sync_scale_from_spin(self, value: int) -> None:
        if self.scale_slider.value() != value:
            self.scale_slider.blockSignals(True)
            self.scale_slider.setValue(value)
            self.scale_slider.blockSignals(False)
        self.update_preview()

    def update_preview(self) -> bytes:
        fmt = self.format_combo.currentText()
        quality = self.quality_slider.value()
        scale = self.scale_slider.value()

        data = encode_image_bytes(
            self._composited,
            fmt=fmt,
            quality=quality,
            scale_percent=scale,
        )
        self.last_encoded_bytes = data

        out_w = max(1, int(round(self.doc.width * scale / 100.0)))
        out_h = max(1, int(round(self.doc.height * scale / 100.0)))
        self.size_label.setText(tr("export.est_size", size=format_byte_size(len(data))))
        self.dimensions_label.setText(f"{out_w} x {out_h} px ({fmt})")

        pix = QPixmap()
        if pix.loadFromData(data):
            if pix.width() > 360 or pix.height() > 220:
                pix = pix.scaled(
                    360,
                    220,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            self.preview_label.setPixmap(pix)
        return data

    def export_to_file(self, path: Optional[Path] = None) -> Optional[Path]:
        fmt = self.format_combo.currentText().upper()
        ext_map = {"PNG": "png", "JPEG": "jpg", "WEBP": "webp"}
        ext = ext_map.get(fmt, "png")

        if path is None:
            default_name = f"{self.doc.title}.{ext}"
            chosen, _ = QFileDialog.getSaveFileName(
                self,
                tr("export.save_title"),
                default_name,
                f"{fmt} Files (*.{ext});;All Files (*)",
            )
            if not chosen:
                return None
            target = Path(chosen)
        else:
            target = Path(path)

        data = self.update_preview()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        self.exported_path = target
        self.accept()
        return target


class ChromaKeyDialog(QDialog):
    """Real-time C-accelerated Chroma Key dialog with cancel restore and Delta Undo commit."""

    preview_updated = Signal()

    def __init__(
        self,
        doc: DocumentSession,
        parent: Optional[QWidget] = None,
        on_preview: Optional[Callable[[], None]] = None,
    ) -> None:
        super().__init__(parent)
        self.doc = doc
        self._on_preview = on_preview
        self._layer = doc.active_layer
        self._original_pixels = (
            self._layer.pixels.copy() if (self._layer is not None and not self._layer.locked) else None
        )
        self._finalized = False

        self.setWindowTitle(tr("chroma.title"))
        self.setMinimumWidth(440)
        self._build_ui()
        self.apply_preview()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(300, 180)
        self.preview_label.setStyleSheet("background: #232328; border: 1px solid #3c3c44; border-radius: 4px;")
        root.addWidget(self.preview_label)

        color_group = QGroupBox(tr("chroma.key_color"))
        color_layout = QHBoxLayout(color_group)

        self.color_btn = QPushButton()
        self.color_btn.setFixedSize(44, 26)
        self.r_spin = QSpinBox()
        self.g_spin = QSpinBox()
        self.b_spin = QSpinBox()
        for spin, val, label_txt in (
            (self.r_spin, 0, "R:"),
            (self.g_spin, 255, "G:"),
            (self.b_spin, 0, "B:"),
        ):
            spin.setRange(0, 255)
            spin.setValue(val)
            color_layout.addWidget(QLabel(label_txt))
            color_layout.addWidget(spin)

        color_layout.addWidget(self.color_btn)
        self.sample_corner_btn = QPushButton(tr("chroma.eyedropper_corner"))
        color_layout.addWidget(self.sample_corner_btn)
        root.addWidget(color_group)

        form = QFormLayout()

        self.tolerance_slider, self.tolerance_spin = self._make_slider_pair(0, 255, 40)
        tol_row = QHBoxLayout()
        tol_row.addWidget(self.tolerance_slider, 1)
        tol_row.addWidget(self.tolerance_spin)
        form.addRow(tr("chroma.tolerance"), tol_row)

        self.softness_slider, self.softness_spin = self._make_slider_pair(0, 255, 20)
        soft_row = QHBoxLayout()
        soft_row.addWidget(self.softness_slider, 1)
        soft_row.addWidget(self.softness_spin)
        form.addRow(tr("chroma.softness"), soft_row)

        self.despill_slider, self.despill_spin = self._make_slider_pair(0, 100, 50)
        self.despill_spin.setSuffix("%")
        desp_row = QHBoxLayout()
        desp_row.addWidget(self.despill_slider, 1)
        desp_row.addWidget(self.despill_spin)
        form.addRow(tr("chroma.despill"), desp_row)

        root.addLayout(form)

        if self._original_pixels is None:
            warn = QLabel(tr("chroma.locked_warn"))
            warn.setStyleSheet("color: #f0a040;")
            root.addWidget(warn)

        buttons = QDialogButtonBox()
        self.apply_btn = buttons.addButton(tr("dlg.apply"), QDialogButtonBox.ButtonRole.AcceptRole)
        self.cancel_btn = buttons.addButton(tr("dlg.cancel"), QDialogButtonBox.ButtonRole.RejectRole)
        self.apply_btn.setEnabled(self._original_pixels is not None)
        root.addWidget(buttons)

        self._update_swatch_style()

        self.color_btn.clicked.connect(self._pick_color_dialog)
        self.sample_corner_btn.clicked.connect(self.sample_top_left_pixel)
        for spin in (self.r_spin, self.g_spin, self.b_spin):
            spin.valueChanged.connect(self._on_rgb_changed)

        self._wire_pair(self.tolerance_slider, self.tolerance_spin)
        self._wire_pair(self.softness_slider, self.softness_spin)
        self._wire_pair(self.despill_slider, self.despill_spin)

        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

    @staticmethod
    def _make_slider_pair(low: int, high: int, initial: int) -> tuple[QSlider, QSpinBox]:
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(low, high)
        slider.setValue(initial)
        spin = QSpinBox()
        spin.setRange(low, high)
        spin.setValue(initial)
        return slider, spin

    def _wire_pair(self, slider: QSlider, spin: QSpinBox) -> None:
        def from_slider(val: int) -> None:
            if spin.value() != val:
                spin.blockSignals(True)
                spin.setValue(val)
                spin.blockSignals(False)
            self.apply_preview()

        def from_spin(val: int) -> None:
            if slider.value() != val:
                slider.blockSignals(True)
                slider.setValue(val)
                slider.blockSignals(False)
            self.apply_preview()

        slider.valueChanged.connect(from_slider)
        spin.valueChanged.connect(from_spin)

    @property
    def key_rgb(self) -> tuple[int, int, int]:
        return (self.r_spin.value(), self.g_spin.value(), self.b_spin.value())

    def set_key_color(self, r: int, g: int, b: int) -> None:
        for spin, val in ((self.r_spin, r), (self.g_spin, g), (self.b_spin, b)):
            spin.blockSignals(True)
            spin.setValue(max(0, min(255, int(val))))
            spin.blockSignals(False)
        self._update_swatch_style()
        self.apply_preview()

    def sample_top_left_pixel(self) -> None:
        if self._original_pixels is None or self._original_pixels.size == 0:
            return
        r, g, b, _ = self._original_pixels[0, 0]
        self.set_key_color(int(r), int(g), int(b))

    def _pick_color_dialog(self) -> None:
        initial = QColor(*self.key_rgb)
        chosen = QColorDialog.getColor(initial, self, tr("chroma.pick_color_title"))
        if chosen.isValid():
            self.set_key_color(chosen.red(), chosen.green(), chosen.blue())

    def _on_rgb_changed(self, _value: int) -> None:
        self._update_swatch_style()
        self.apply_preview()

    def _update_swatch_style(self) -> None:
        r, g, b = self.key_rgb
        self.color_btn.setStyleSheet(
            f"background-color: rgb({r}, {g}, {b}); border: 1px solid #888; border-radius: 3px;"
        )

    def _notify_preview(self) -> None:
        self.preview_updated.emit()
        if self._on_preview is not None:
            self._on_preview()

    def apply_preview(self) -> None:
        if self._layer is None or self._original_pixels is None:
            return
        np.copyto(self._layer.pixels, self._original_pixels)
        get_engine().chroma_key(
            self._layer.pixels,
            key_rgb=self.key_rgb,
            tolerance=float(self.tolerance_slider.value()),
            softness=float(self.softness_slider.value()),
            despill=float(self.despill_slider.value()) / 100.0,
        )
        self.doc.invalidate_composite()
        self.preview_label.setPixmap(_rgba_to_qpixmap(self._layer.pixels, max_w=320, max_h=180))
        self._notify_preview()

    def _restore_original(self) -> None:
        if self._layer is not None and self._original_pixels is not None:
            np.copyto(self._layer.pixels, self._original_pixels)
            self.doc.invalidate_composite()
            self._notify_preview()

    def accept(self) -> None:
        if self._finalized:
            return
        self._finalized = True
        if self._layer is not None and self._original_pixels is not None:
            self.apply_preview()
            dirty_rect = compute_dirty_rect(self._original_pixels, self._layer.pixels)
            if dirty_rect is not None:
                rx, ry, rw, rh = dirty_rect
                before_patch = np.ascontiguousarray(
                    self._original_pixels[ry : ry + rh, rx : rx + rw].copy()
                )
                after_patch = self._layer.get_patch(dirty_rect)
                self.doc.record_delta_patch(
                    layer_id=self._layer.id,
                    dirty_rect=dirty_rect,
                    before_patch=before_patch,
                    after_patch=after_patch,
                    target="pixels",
                    description="Chroma Key",
                )
        super().accept()

    def reject(self) -> None:
        if not self._finalized:
            self._finalized = True
            self._restore_original()
        super().reject()

    def closeEvent(self, event) -> None:
        if not self._finalized:
            self._finalized = True
            self._restore_original()
        super().closeEvent(event)


class GridSplitterDialog(QDialog):
    """Dialog to slice canvas or active layer into a Rows x Cols grid or fixed pixel tiles."""

    tabs_opened = Signal(list)
    exported = Signal(list)

    def __init__(
        self,
        doc: DocumentSession,
        tab_manager: DocumentTabManager,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.doc = doc
        self.tab_manager = tab_manager

        self.setWindowTitle(tr("grid.title"))
        self.setMinimumWidth(400)
        self._build_ui()
        self._update_summary()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        form = QFormLayout()
        self.source_combo = QComboBox()
        self.source_combo.addItem(tr("dlg.source_canvas"), "canvas")
        self.source_combo.addItem(tr("dlg.source_layer"), "layer")
        form.addRow(tr("dlg.source"), self.source_combo)

        mode_box = QGroupBox(tr("grid.mode_box"))
        mode_layout = QGridLayout(mode_box)

        self.grid_radio = QRadioButton(tr("grid.rows_cols"))
        self.grid_radio.setChecked(True)
        self.rows_spin = QSpinBox()
        self.rows_spin.setRange(1, 256)
        self.rows_spin.setValue(2)
        self.cols_spin = QSpinBox()
        self.cols_spin.setRange(1, 256)
        self.cols_spin.setValue(2)

        self.tile_radio = QRadioButton(tr("grid.fixed_tile"))
        self.tile_w_spin = QSpinBox()
        self.tile_w_spin.setRange(1, 8192)
        self.tile_w_spin.setValue(64)
        self.tile_h_spin = QSpinBox()
        self.tile_h_spin.setRange(1, 8192)
        self.tile_h_spin.setValue(64)

        mode_layout.addWidget(self.grid_radio, 0, 0, 1, 4)
        mode_layout.addWidget(QLabel(tr("grid.rows")), 1, 0)
        mode_layout.addWidget(self.rows_spin, 1, 1)
        mode_layout.addWidget(QLabel(tr("grid.cols")), 1, 2)
        mode_layout.addWidget(self.cols_spin, 1, 3)

        mode_layout.addWidget(self.tile_radio, 2, 0, 1, 4)
        mode_layout.addWidget(QLabel(tr("grid.tile_w")), 3, 0)
        mode_layout.addWidget(self.tile_w_spin, 3, 1)
        mode_layout.addWidget(QLabel(tr("grid.tile_h")), 3, 2)
        mode_layout.addWidget(self.tile_h_spin, 3, 3)

        root.addLayout(form)
        root.addWidget(mode_box)

        prefix_form = QFormLayout()
        self.prefix_edit = QLineEdit("tile")
        prefix_form.addRow(tr("grid.prefix"), self.prefix_edit)
        root.addLayout(prefix_form)

        self.count_label = QLabel("")
        self.count_label.setStyleSheet("font-weight: 600;")
        root.addWidget(self.count_label)

        btn_row = QHBoxLayout()
        self.export_btn = QPushButton(tr("dlg.export_pngs"))
        self.open_tabs_btn = QPushButton(tr("dlg.open_tabs"))
        self.close_btn = QPushButton(tr("dlg.close"))
        btn_row.addWidget(self.export_btn)
        btn_row.addWidget(self.open_tabs_btn)
        btn_row.addWidget(self.close_btn)
        root.addLayout(btn_row)

        for sig in (
            self.source_combo.currentIndexChanged,
            self.grid_radio.toggled,
            self.tile_radio.toggled,
            self.rows_spin.valueChanged,
            self.cols_spin.valueChanged,
            self.tile_w_spin.valueChanged,
            self.tile_h_spin.valueChanged,
        ):
            sig.connect(self._update_summary)

        self.export_btn.clicked.connect(self._on_export_clicked)
        self.open_tabs_btn.clicked.connect(self._on_open_tabs_clicked)
        self.close_btn.clicked.connect(self.reject)

    def _source_rgba(self) -> np.ndarray:
        if self.source_combo.currentData() == "layer" and self.doc.active_layer is not None:
            return np.ascontiguousarray(self.doc.active_layer.pixels, dtype=np.uint8)
        return np.ascontiguousarray(self.doc.composite(), dtype=np.uint8)

    def _update_summary(self) -> None:
        use_tiles = self.tile_radio.isChecked()
        self.rows_spin.setEnabled(not use_tiles)
        self.cols_spin.setEnabled(not use_tiles)
        self.tile_w_spin.setEnabled(use_tiles)
        self.tile_h_spin.setEnabled(use_tiles)

        buf = self._source_rgba()
        h, w = int(buf.shape[0]), int(buf.shape[1])
        if use_tiles:
            tw = self.tile_w_spin.value()
            th = self.tile_h_spin.value()
            cols = (w + tw - 1) // tw
            rows = (h + th - 1) // th
            total = cols * rows
        else:
            rows = min(self.rows_spin.value(), h)
            cols = min(self.cols_spin.value(), w)
            total = rows * cols
        self.count_label.setText(tr("grid.summary", total=total, w=w, h=h))

    def compute_slices(self) -> list[tuple[tuple[int, int, int, int], np.ndarray]]:
        rgba = self._source_rgba()
        if self.tile_radio.isChecked():
            return split_grid(
                rgba,
                tile_w=self.tile_w_spin.value(),
                tile_h=self.tile_h_spin.value(),
            )
        return split_grid(
            rgba,
            rows=self.rows_spin.value(),
            cols=self.cols_spin.value(),
        )

    def export_to_folder(self, output_dir: Path) -> list[Path]:
        slices = self.compute_slices()
        prefix = self.prefix_edit.text().strip() or "tile"
        paths = export_slices_to_folder(slices, output_dir=Path(output_dir), prefix=prefix)
        self.exported.emit(paths)
        return paths

    def open_in_tabs(self) -> list[DocumentSession]:
        slices = self.compute_slices()
        prefix = self.prefix_edit.text().strip() or "Tile"
        docs = open_slices_as_tabs(slices, tab_manager=self.tab_manager, prefix=prefix)
        self.tabs_opened.emit(docs)
        return docs

    def _on_export_clicked(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, tr("dlg.select_folder"))
        if not folder:
            return
        paths = self.export_to_folder(Path(folder))
        QMessageBox.information(
            self,
            tr("grid.title"),
            tr("grid.exported_msg", count=len(paths), folder=folder),
        )

    def _on_open_tabs_clicked(self) -> None:
        docs = self.open_in_tabs()
        if docs:
            self.accept()


class SmartAlphaSplitterDialog(QDialog):
    """C-accelerated Connected-Component Sprite Splitter dialog with canvas box preview."""

    preview_boxes_changed = Signal(list)
    tabs_opened = Signal(list)
    exported = Signal(list)

    def __init__(
        self,
        doc: DocumentSession,
        tab_manager: DocumentTabManager,
        on_preview_boxes: Optional[Callable[[list[ComponentBox]], None]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.doc = doc
        self.tab_manager = tab_manager
        self._on_preview_boxes = on_preview_boxes
        self.detected_boxes: list[ComponentBox] = []

        self.setWindowTitle(tr("alpha_split.title"))
        self.setMinimumWidth(420)
        self._build_ui()
        self.preview_on_canvas()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        form = QFormLayout()
        self.source_combo = QComboBox()
        self.source_combo.addItem(tr("dlg.source_layer"), "layer")
        self.source_combo.addItem(tr("dlg.source_canvas"), "canvas")
        form.addRow(tr("dlg.source"), self.source_combo)

        thr_row = QHBoxLayout()
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(0, 254)
        self.threshold_slider.setValue(10)
        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(0, 254)
        self.threshold_spin.setValue(10)
        thr_row.addWidget(self.threshold_slider, 1)
        thr_row.addWidget(self.threshold_spin)
        form.addRow(tr("alpha_split.threshold"), thr_row)

        self.min_area_spin = QSpinBox()
        self.min_area_spin.setRange(1, 10_000_000)
        self.min_area_spin.setValue(16)
        form.addRow(tr("alpha_split.min_area"), self.min_area_spin)

        self.padding_spin = QSpinBox()
        self.padding_spin.setRange(0, 256)
        self.padding_spin.setValue(2)
        form.addRow(tr("alpha_split.padding"), self.padding_spin)

        self.prefix_edit = QLineEdit("sprite")
        form.addRow(tr("alpha_split.prefix"), self.prefix_edit)
        root.addLayout(form)

        self.preview_checkbox = QCheckBox(tr("alpha_split.auto_preview"))
        self.preview_checkbox.setChecked(True)
        root.addWidget(self.preview_checkbox)

        self.count_label = QLabel(tr("alpha_split.detected", count=0))
        self.count_label.setStyleSheet("font-weight: 600;")
        root.addWidget(self.count_label)

        btn_row = QHBoxLayout()
        self.preview_btn = QPushButton(tr("alpha_split.preview_btn"))
        self.export_btn = QPushButton(tr("dlg.export_pngs"))
        self.open_tabs_btn = QPushButton(tr("dlg.open_tabs"))
        self.close_btn = QPushButton(tr("dlg.close"))
        btn_row.addWidget(self.preview_btn)
        btn_row.addWidget(self.export_btn)
        btn_row.addWidget(self.open_tabs_btn)
        btn_row.addWidget(self.close_btn)
        root.addLayout(btn_row)

        self.threshold_slider.valueChanged.connect(self._sync_threshold_from_slider)
        self.threshold_spin.valueChanged.connect(self._sync_threshold_from_spin)
        self.min_area_spin.valueChanged.connect(self._on_param_changed)
        self.padding_spin.valueChanged.connect(self._on_param_changed)
        self.source_combo.currentIndexChanged.connect(self._on_param_changed)

        self.preview_btn.clicked.connect(self.preview_on_canvas)
        self.export_btn.clicked.connect(self._on_export_clicked)
        self.open_tabs_btn.clicked.connect(self._on_open_tabs_clicked)
        self.close_btn.clicked.connect(self.reject)

    def _sync_threshold_from_slider(self, value: int) -> None:
        if self.threshold_spin.value() != value:
            self.threshold_spin.blockSignals(True)
            self.threshold_spin.setValue(value)
            self.threshold_spin.blockSignals(False)
        self._on_param_changed()

    def _sync_threshold_from_spin(self, value: int) -> None:
        if self.threshold_slider.value() != value:
            self.threshold_slider.blockSignals(True)
            self.threshold_slider.setValue(value)
            self.threshold_slider.blockSignals(False)
        self._on_param_changed()

    def _on_param_changed(self) -> None:
        if self.preview_checkbox.isChecked():
            self.preview_on_canvas()
        else:
            boxes = self.detect_boxes()
            self.count_label.setText(tr("alpha_split.detected", count=len(boxes)))

    def _source_rgba(self) -> np.ndarray:
        if self.source_combo.currentData() == "layer" and self.doc.active_layer is not None:
            return np.ascontiguousarray(self.doc.active_layer.pixels, dtype=np.uint8)
        return np.ascontiguousarray(self.doc.composite(), dtype=np.uint8)

    def detect_boxes(self) -> list[ComponentBox]:
        rgba = self._source_rgba()
        boxes = detect_alpha_sprites(
            rgba,
            alpha_threshold=self.threshold_spin.value(),
            min_area=self.min_area_spin.value(),
            padding=self.padding_spin.value(),
        )
        self.detected_boxes = boxes
        self.count_label.setText(tr("alpha_split.detected", count=len(boxes)))
        return boxes

    def _emit_boxes(self, boxes: list[ComponentBox]) -> None:
        self.preview_boxes_changed.emit(boxes)
        if self._on_preview_boxes is not None:
            self._on_preview_boxes(boxes)

    def preview_on_canvas(self) -> list[ComponentBox]:
        boxes = self.detect_boxes()
        self._emit_boxes(boxes)
        return boxes

    def extract_slices(self) -> list[tuple[ComponentBox, np.ndarray]]:
        rgba = self._source_rgba()
        boxes = self.detect_boxes()
        return extract_alpha_sprites(rgba, boxes)

    def export_to_folder(self, output_dir: Path) -> list[Path]:
        slices = self.extract_slices()
        prefix = self.prefix_edit.text().strip() or "sprite"
        paths = export_slices_to_folder(slices, output_dir=Path(output_dir), prefix=prefix)
        self.exported.emit(paths)
        return paths

    def open_in_tabs(self) -> list[DocumentSession]:
        slices = self.extract_slices()
        prefix = self.prefix_edit.text().strip() or "Sprite"
        docs = open_slices_as_tabs(slices, tab_manager=self.tab_manager, prefix=prefix)
        self.tabs_opened.emit(docs)
        return docs

    def _on_export_clicked(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, tr("dlg.select_folder"))
        if not folder:
            return
        paths = self.export_to_folder(Path(folder))
        QMessageBox.information(
            self,
            tr("alpha_split.title"),
            tr("alpha_split.exported_msg", count=len(paths), folder=folder),
        )

    def _on_open_tabs_clicked(self) -> None:
        docs = self.open_in_tabs()
        if docs:
            self._emit_boxes([])
            self.accept()

    def reject(self) -> None:
        self._emit_boxes([])
        super().reject()

    def closeEvent(self, event) -> None:
        self._emit_boxes([])
        super().closeEvent(event)


class _BatchWorker(QObject):
    progress_updated = Signal(int, int, str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, config: BatchOperationConfig, cancel_event: threading.Event) -> None:
        super().__init__()
        self.config = config
        self.cancel_event = cancel_event

    def run(self) -> None:
        try:
            processor = BatchProcessor(self.config)
            result = processor.run(
                progress_callback=lambda done, total, name: self.progress_updated.emit(done, total, name),
                cancel_event=self.cancel_event,
            )
            self.finished.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class BatchProcessorDialog(QDialog):
    """UI dialog configuring and running multi-threaded BatchProcessor pipelines."""

    batch_completed = Signal(object)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.last_result: Optional[BatchResult] = None
        self._thread: Optional[QThread] = None
        self._worker: Optional[_BatchWorker] = None
        self._cancel_event: Optional[threading.Event] = None

        self.setWindowTitle(tr("batch.title"))
        self.setMinimumWidth(520)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        form = QFormLayout()

        in_row = QHBoxLayout()
        self.input_dir_edit = QLineEdit()
        self.input_browse_btn = QPushButton(tr("dlg.browse"))
        in_row.addWidget(self.input_dir_edit, 1)
        in_row.addWidget(self.input_browse_btn)
        form.addRow(tr("batch.in_dir"), in_row)

        out_row = QHBoxLayout()
        self.output_dir_edit = QLineEdit()
        self.output_browse_btn = QPushButton(tr("dlg.browse"))
        out_row.addWidget(self.output_dir_edit, 1)
        out_row.addWidget(self.output_browse_btn)
        form.addRow(tr("batch.out_dir"), out_row)

        fmt_row = QHBoxLayout()
        self.format_combo = QComboBox()
        self.format_combo.addItems(["PNG", "JPEG", "WEBP"])
        self.quality_spin = QSpinBox()
        self.quality_spin.setRange(1, 100)
        self.quality_spin.setValue(90)
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(1, 16)
        self.workers_spin.setValue(4)
        fmt_row.addWidget(self.format_combo)
        fmt_row.addWidget(QLabel(tr("export.quality")))
        fmt_row.addWidget(self.quality_spin)
        fmt_row.addWidget(QLabel(tr("batch.threads")))
        fmt_row.addWidget(self.workers_spin)
        form.addRow(tr("batch.out_settings"), fmt_row)
        root.addLayout(form)

        ops_box = QGroupBox(tr("batch.ops_box"))
        ops_layout = QVBoxLayout(ops_box)

        self.ai_remove_bg_check = QCheckBox(tr("batch.op_ai_bg"))
        ops_layout.addWidget(self.ai_remove_bg_check)

        chroma_row = QHBoxLayout()
        self.chroma_key_check = QCheckBox(tr("batch.op_chroma"))
        self.chroma_r_spin = QSpinBox()
        self.chroma_g_spin = QSpinBox()
        self.chroma_b_spin = QSpinBox()
        for sp, val in ((self.chroma_r_spin, 0), (self.chroma_g_spin, 255), (self.chroma_b_spin, 0)):
            sp.setRange(0, 255)
            sp.setValue(val)
        self.chroma_tol_spin = QDoubleSpinBox()
        self.chroma_tol_spin.setRange(0.0, 255.0)
        self.chroma_tol_spin.setValue(40.0)
        chroma_row.addWidget(self.chroma_key_check)
        chroma_row.addWidget(QLabel("RGB:"))
        chroma_row.addWidget(self.chroma_r_spin)
        chroma_row.addWidget(self.chroma_g_spin)
        chroma_row.addWidget(self.chroma_b_spin)
        chroma_row.addWidget(QLabel(tr("batch.op_tol")))
        chroma_row.addWidget(self.chroma_tol_spin)
        ops_layout.addLayout(chroma_row)

        self.trim_check = QCheckBox(tr("batch.op_trim"))
        ops_layout.addWidget(self.trim_check)

        resize_row = QHBoxLayout()
        self.resize_check = QCheckBox(tr("batch.op_resize"))
        self.resize_w_spin = QSpinBox()
        self.resize_w_spin.setRange(1, 16384)
        self.resize_w_spin.setValue(512)
        self.resize_h_spin = QSpinBox()
        self.resize_h_spin.setRange(1, 16384)
        self.resize_h_spin.setValue(512)
        resize_row.addWidget(self.resize_check)
        resize_row.addWidget(QLabel(tr("batch.w")))
        resize_row.addWidget(self.resize_w_spin)
        resize_row.addWidget(QLabel(tr("batch.h")))
        resize_row.addWidget(self.resize_h_spin)
        resize_row.addStretch(1)
        ops_layout.addLayout(resize_row)

        root.addWidget(ops_box)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.status_label = QLabel(tr("batch.ready"))
        root.addWidget(self.progress_bar)
        root.addWidget(self.status_label)

        btn_row = QHBoxLayout()
        self.run_btn = QPushButton(tr("batch.run"))
        self.close_btn = QPushButton(tr("dlg.close"))
        btn_row.addStretch(1)
        btn_row.addWidget(self.run_btn)
        btn_row.addWidget(self.close_btn)
        root.addLayout(btn_row)

        self.input_browse_btn.clicked.connect(self._browse_input)
        self.output_browse_btn.clicked.connect(self._browse_output)
        self.run_btn.clicked.connect(self.start_batch)
        self.close_btn.clicked.connect(self.reject)

    def _browse_input(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, tr("batch.sel_in"))
        if folder:
            self.input_dir_edit.setText(folder)

    def _browse_output(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, tr("batch.sel_out"))
        if folder:
            self.output_dir_edit.setText(folder)

    def build_config(self) -> BatchOperationConfig:
        in_dir = Path(self.input_dir_edit.text().strip())
        out_dir = Path(self.output_dir_edit.text().strip())
        resize_w = self.resize_w_spin.value() if self.resize_check.isChecked() else None
        resize_h = self.resize_h_spin.value() if self.resize_check.isChecked() else None
        return BatchOperationConfig(
            input_dir=in_dir,
            output_dir=out_dir,
            output_format=self.format_combo.currentText(),
            quality=self.quality_spin.value(),
            ai_remove_bg=self.ai_remove_bg_check.isChecked(),
            chroma_key_enabled=self.chroma_key_check.isChecked(),
            chroma_key_rgb=(
                self.chroma_r_spin.value(),
                self.chroma_g_spin.value(),
                self.chroma_b_spin.value(),
            ),
            chroma_tolerance=float(self.chroma_tol_spin.value()),
            trim_transparent=self.trim_check.isChecked(),
            resize_width=resize_w,
            resize_height=resize_h,
            max_workers=self.workers_spin.value(),
        )

    def run_batch_sync(self) -> BatchResult:
        cfg = self.build_config()
        processor = BatchProcessor(cfg)

        def on_progress(done: int, total: int, name: str) -> None:
            pct = int(round((done / max(total, 1)) * 100))
            self.progress_bar.setValue(pct)
            self.status_label.setText(f"[{done}/{total}] {name}")

        result = processor.run(progress_callback=on_progress)
        self.last_result = result
        self.status_label.setText(
            tr("batch.done", ok=result.succeeded, total=result.total, fail=result.failed)
        )
        self.batch_completed.emit(result)
        return result

    def start_batch(self) -> None:
        in_text = self.input_dir_edit.text().strip()
        out_text = self.output_dir_edit.text().strip()
        if not in_text or not out_text:
            QMessageBox.warning(self, tr("batch.title"), tr("batch.warn_dirs"))
            return
        cfg = self.build_config()
        if not cfg.input_dir.is_dir():
            QMessageBox.warning(
                self,
                tr("batch.title"),
                tr("batch.warn_missing_in", path=cfg.input_dir),
            )
            return

        self.run_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText(tr("batch.starting"))

        self._cancel_event = threading.Event()
        self._thread = QThread(self)
        self._worker = _BatchWorker(cfg, self._cancel_event)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progress_updated.connect(self._on_worker_progress)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.failed.connect(self._on_worker_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_worker)
        self._thread.start()

    def _on_worker_progress(self, done: int, total: int, name: str) -> None:
        pct = int(round((done / max(total, 1)) * 100))
        self.progress_bar.setValue(pct)
        self.status_label.setText(f"[{done}/{total}] {name}")

    def _on_worker_finished(self, result: BatchResult) -> None:
        self.last_result = result
        self.run_btn.setEnabled(True)
        self.progress_bar.setValue(100 if result.total > 0 else 0)
        self.status_label.setText(
            tr("batch.done", ok=result.succeeded, total=result.total, fail=result.failed)
        )
        self.batch_completed.emit(result)

    def _on_worker_failed(self, msg: str) -> None:
        self.run_btn.setEnabled(True)
        self.status_label.setText(f"{tr('batch.error')}: {msg}")
        QMessageBox.critical(self, tr("batch.error"), msg)

    def _cleanup_worker(self) -> None:
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None
        if self._thread is not None:
            self._thread.deleteLater()
            self._thread = None

    def _stop_thread_if_running(self) -> None:
        if self._cancel_event is not None:
            self._cancel_event.set()
        if self._thread is not None and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait(5000)

    def reject(self) -> None:
        self._stop_thread_if_running()
        super().reject()

    def closeEvent(self, event) -> None:
        self._stop_thread_if_running()
        super().closeEvent(event)


class _HotkeySequenceEdit(QKeySequenceEdit):
    """QKeySequenceEdit that also captures the Tab key cleanly."""

    def event(self, event: QEvent) -> bool:
        if event.type() == QEvent.Type.KeyPress:
            key = event.key()  # type: ignore[attr-defined]
            if key == Qt.Key.Key_Tab:
                self.setKeySequence(QKeySequence(Qt.Key.Key_Tab))
                event.accept()
                return True
        return super().event(event)


class HotkeyEditorDialog(QDialog):
    """Dialog for inspecting, editing, resetting, and saving app/config/keybindings.json."""

    keybindings_saved = Signal(dict)

    def __init__(
        self,
        keybindings_path: Optional[Path] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.keybindings_path = Path(keybindings_path) if keybindings_path is not None else None
        self.bindings = load_keybindings(self.keybindings_path)
        self._editors: dict[str, _HotkeySequenceEdit] = {}

        self.setWindowTitle(tr("hotkeys.title"))
        self.setMinimumSize(460, 520)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        self.table = QTableWidget(len(self.bindings), 2)
        self.table.setHorizontalHeaderLabels([tr("hotkeys.col_action"), tr("hotkeys.col_shortcut")])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)

        for row, (action, shortcut) in enumerate(self.bindings.items()):
            label = tr(f"hk.{action}")
            if label == f"hk.{action}":
                label = ACTION_DISPLAY_NAMES.get(action, action)
            item = QTableWidgetItem(label)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            item.setData(Qt.ItemDataRole.UserRole, action)
            self.table.setItem(row, 0, item)

            editor = _HotkeySequenceEdit(QKeySequence.fromString(shortcut))
            self._editors[action] = editor
            self.table.setCellWidget(row, 1, editor)

        root.addWidget(self.table, 1)

        btn_row = QHBoxLayout()
        self.reset_btn = QPushButton(tr("hotkeys.reset"))
        btn_row.addWidget(self.reset_btn)
        btn_row.addStretch(1)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        self.save_btn = buttons.button(QDialogButtonBox.StandardButton.Save)
        btn_row.addWidget(buttons)
        root.addLayout(btn_row)

        self.reset_btn.clicked.connect(self.reset_defaults)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

    def get_bindings(self) -> dict[str, str]:
        result: dict[str, str] = {}
        for action, editor in self._editors.items():
            seq = editor.keySequence().toString(QKeySequence.SequenceFormat.PortableText).strip()
            result[action] = seq
        return result

    def set_binding(self, action: str, shortcut: str) -> None:
        if action not in self._editors:
            raise KeyError(f"Unknown keybinding action: {action!r}")
        self._editors[action].setKeySequence(QKeySequence.fromString(shortcut))
        self.bindings[action] = shortcut

    def reset_defaults(self) -> None:
        for action, default_seq in DEFAULT_KEYBINDINGS.items():
            if action in self._editors:
                self._editors[action].setKeySequence(QKeySequence.fromString(default_seq))
        self.bindings = dict(DEFAULT_KEYBINDINGS)

    def save(self) -> Path:
        self.bindings = self.get_bindings()
        saved_path = save_keybindings(self.bindings, path=self.keybindings_path)
        self.keybindings_saved.emit(dict(self.bindings))
        return saved_path

    def accept(self) -> None:
        self.save()
        super().accept()
