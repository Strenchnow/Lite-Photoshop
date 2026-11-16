"""Dockable and toolbar panels: ToolOptionsBar, ColorPanel, and LayersPanel."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QColor, QKeyEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app.core.document import CallableCommand, DocumentSession
from app.core.layer import BLEND_MODE_LABELS, BlendMode, Layer, LayerMask
from app.tools.base_tool import ToolContext


_PALETTE_SWATCHES: tuple[tuple[int, int, int], ...] = (
    (0, 0, 0),
    (64, 64, 64),
    (128, 128, 128),
    (192, 192, 192),
    (255, 255, 255),
    (239, 68, 68),
    (249, 115, 22),
    (234, 179, 8),
    (34, 197, 94),
    (20, 184, 166),
    (59, 130, 246),
    (99, 102, 241),
    (168, 85, 247),
    (236, 72, 153),
    (139, 92, 246),
    (0, 229, 255),
)

_COMMON_FONTS: tuple[str, ...] = (
    "Segoe UI",
    "Arial",
    "Consolas",
    "Courier New",
    "Georgia",
    "Tahoma",
    "Times New Roman",
    "Trebuchet MS",
    "Verdana",
)


def _rgba_to_qcolor(rgba: tuple[int, int, int, int]) -> QColor:
    r, g, b, a = rgba
    return QColor(int(r), int(g), int(b), int(a))


def _qcolor_to_rgba(c: QColor) -> tuple[int, int, int, int]:
    return (c.red(), c.green(), c.blue(), c.alpha())


class ToolOptionsBar(QWidget):
    """Horizontal dynamic options bar adapting to the active tool."""

    optionsChanged = Signal()
    centerRequested = Signal(bool, bool)
    clearSelectionRequested = Signal()
    fitViewRequested = Signal()
    zoom100Requested = Signal()
    toolActionRequested = Signal(str)

    def __init__(self, tool_context: ToolContext, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("ToolOptionsBar")
        self.setFixedHeight(40)
        self.context = tool_context
        self._current_tool_id: str = "brush"
        self._syncing: bool = False

        root = QHBoxLayout(self)
        root.setContentsMargins(8, 4, 8, 4)
        root.setSpacing(8)

        self.tool_name_label = QLabel("Brush")
        self.tool_name_label.setStyleSheet("font-weight: 600; color: #ffffff; min-width: 90px;")
        root.addWidget(self.tool_name_label)

        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet("color: #3e3e42;")
        root.addWidget(sep)

        self.stack = QStackedWidget()
        root.addWidget(self.stack, 1)

        self._page_indices: dict[str, int] = {}
        self._build_pages()
        self.sync_from_context()
        self.set_active_tool("brush")

    def _add_page(self, key: str, widget: QWidget) -> None:
        idx = self.stack.addWidget(widget)
        self._page_indices[key] = idx

    def _build_pages(self) -> None:
        # 1. Brush / Eraser page
        brush_w = QWidget()
        bl = QHBoxLayout(brush_w)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(8)

        bl.addWidget(QLabel("Size:"))
        self.brush_radius_spin = QDoubleSpinBox()
        self.brush_radius_spin.setRange(1.0, 250.0)
        self.brush_radius_spin.setSingleStep(1.0)
        self.brush_radius_spin.setSuffix(" px")
        self.brush_radius_spin.valueChanged.connect(self._on_brush_changed)
        bl.addWidget(self.brush_radius_spin)

        bl.addWidget(QLabel("Hardness:"))
        self.brush_hardness_slider = QSlider(Qt.Horizontal)
        self.brush_hardness_slider.setRange(0, 100)
        self.brush_hardness_slider.setFixedWidth(85)
        self.brush_hardness_slider.valueChanged.connect(self._on_brush_changed)
        bl.addWidget(self.brush_hardness_slider)
        self.brush_hardness_val = QLabel("80%")
        self.brush_hardness_val.setFixedWidth(34)
        bl.addWidget(self.brush_hardness_val)

        bl.addWidget(QLabel("Opacity:"))
        self.brush_opacity_slider = QSlider(Qt.Horizontal)
        self.brush_opacity_slider.setRange(1, 100)
        self.brush_opacity_slider.setFixedWidth(85)
        self.brush_opacity_slider.valueChanged.connect(self._on_brush_changed)
        bl.addWidget(self.brush_opacity_slider)
        self.brush_opacity_val = QLabel("100%")
        self.brush_opacity_val.setFixedWidth(36)
        bl.addWidget(self.brush_opacity_val)

        bl.addWidget(QLabel("Flow:"))
        self.brush_flow_slider = QSlider(Qt.Horizontal)
        self.brush_flow_slider.setRange(1, 100)
        self.brush_flow_slider.setFixedWidth(85)
        self.brush_flow_slider.valueChanged.connect(self._on_brush_changed)
        bl.addWidget(self.brush_flow_slider)
        self.brush_flow_val = QLabel("100%")
        self.brush_flow_val.setFixedWidth(36)
        bl.addWidget(self.brush_flow_val)

        self.edit_mask_chk = QCheckBox("Paint Mask")
        self.edit_mask_chk.toggled.connect(self._on_brush_changed)
        bl.addWidget(self.edit_mask_chk)
        bl.addStretch(1)
        self._add_page("brush", brush_w)

        # 2. Blur / Sharpen page
        filter_w = QWidget()
        fl = QHBoxLayout(filter_w)
        fl.setContentsMargins(0, 0, 0, 0)
        fl.setSpacing(8)

        fl.addWidget(QLabel("Mode:"))
        self.filter_mode_combo = QComboBox()
        self.filter_mode_combo.addItem("Blur", 0)
        self.filter_mode_combo.addItem("Sharpen", 1)
        self.filter_mode_combo.currentIndexChanged.connect(self._on_filter_changed)
        fl.addWidget(self.filter_mode_combo)

        fl.addWidget(QLabel("Kernel:"))
        self.filter_kernel_combo = QComboBox()
        self.filter_kernel_combo.addItem("3x3", 3)
        self.filter_kernel_combo.addItem("5x5", 5)
        self.filter_kernel_combo.currentIndexChanged.connect(self._on_filter_changed)
        fl.addWidget(self.filter_kernel_combo)

        fl.addWidget(QLabel("Size:"))
        self.filter_radius_spin = QDoubleSpinBox()
        self.filter_radius_spin.setRange(1.0, 200.0)
        self.filter_radius_spin.setSuffix(" px")
        self.filter_radius_spin.valueChanged.connect(self._on_filter_changed)
        fl.addWidget(self.filter_radius_spin)

        fl.addWidget(QLabel("Strength:"))
        self.filter_strength_slider = QSlider(Qt.Horizontal)
        self.filter_strength_slider.setRange(1, 100)
        self.filter_strength_slider.setFixedWidth(95)
        self.filter_strength_slider.valueChanged.connect(self._on_filter_changed)
        fl.addWidget(self.filter_strength_slider)
        self.filter_strength_val = QLabel("60%")
        self.filter_strength_val.setFixedWidth(36)
        fl.addWidget(self.filter_strength_val)
        fl.addStretch(1)
        self._add_page("blur_sharpen", filter_w)

        # 3. Smudge page
        smudge_w = QWidget()
        sl = QHBoxLayout(smudge_w)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(8)

        sl.addWidget(QLabel("Size:"))
        self.smudge_radius_spin = QDoubleSpinBox()
        self.smudge_radius_spin.setRange(1.0, 200.0)
        self.smudge_radius_spin.setSuffix(" px")
        self.smudge_radius_spin.valueChanged.connect(self._on_smudge_changed)
        sl.addWidget(self.smudge_radius_spin)

        sl.addWidget(QLabel("Strength:"))
        self.smudge_strength_slider = QSlider(Qt.Horizontal)
        self.smudge_strength_slider.setRange(1, 100)
        self.smudge_strength_slider.setFixedWidth(110)
        self.smudge_strength_slider.valueChanged.connect(self._on_smudge_changed)
        sl.addWidget(self.smudge_strength_slider)
        self.smudge_strength_val = QLabel("70%")
        self.smudge_strength_val.setFixedWidth(36)
        sl.addWidget(self.smudge_strength_val)
        sl.addStretch(1)
        self._add_page("smudge", smudge_w)

        # 4. Magic Wand / Bucket Fill tolerance page
        tol_w = QWidget()
        tl = QHBoxLayout(tol_w)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.setSpacing(8)

        tl.addWidget(QLabel("Tolerance:"))
        self.tolerance_spin = QSpinBox()
        self.tolerance_spin.setRange(0, 255)
        self.tolerance_spin.valueChanged.connect(self._on_tolerance_changed)
        tl.addWidget(self.tolerance_spin)

        self.contiguous_chk = QCheckBox("Contiguous")
        self.contiguous_chk.toggled.connect(self._on_tolerance_changed)
        tl.addWidget(self.contiguous_chk)

        self.tol_clear_sel_btn = QPushButton("Clear Selection")
        self.tol_clear_sel_btn.clicked.connect(self.clearSelectionRequested.emit)
        tl.addWidget(self.tol_clear_sel_btn)
        tl.addStretch(1)
        self._add_page("tolerance", tol_w)

        # 5. Text page
        text_w = QWidget()
        txl = QHBoxLayout(text_w)
        txl.setContentsMargins(0, 0, 0, 0)
        txl.setSpacing(8)

        txl.addWidget(QLabel("Font:"))
        self.font_combo = QComboBox()
        self.font_combo.setEditable(True)
        self.font_combo.addItems(_COMMON_FONTS)
        self.font_combo.setFixedWidth(135)
        self.font_combo.currentTextChanged.connect(self._on_text_changed)
        txl.addWidget(self.font_combo)

        txl.addWidget(QLabel("Size:"))
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(6, 256)
        self.font_size_spin.setSuffix(" pt")
        self.font_size_spin.valueChanged.connect(self._on_text_changed)
        txl.addWidget(self.font_size_spin)

        self.font_bold_chk = QCheckBox("Bold")
        self.font_bold_chk.toggled.connect(self._on_text_changed)
        txl.addWidget(self.font_bold_chk)

        txl.addWidget(QLabel("Text:"))
        self.text_content_edit = QLineEdit()
        self.text_content_edit.setMinimumWidth(140)
        self.text_content_edit.setMaximumWidth(240)
        self.text_content_edit.textChanged.connect(self._on_text_changed)
        txl.addWidget(self.text_content_edit)
        txl.addStretch(1)
        self._add_page("text", text_w)

        # 6. Move page
        move_w = QWidget()
        ml = QHBoxLayout(move_w)
        ml.setContentsMargins(0, 0, 0, 0)
        ml.setSpacing(6)

        self.btn_center_h = QPushButton("Center H")
        self.btn_center_h.clicked.connect(lambda: self.centerRequested.emit(True, False))
        ml.addWidget(self.btn_center_h)

        self.btn_center_v = QPushButton("Center V")
        self.btn_center_v.clicked.connect(lambda: self.centerRequested.emit(False, True))
        ml.addWidget(self.btn_center_v)

        self.btn_center_both = QPushButton("Center Both")
        self.btn_center_both.clicked.connect(lambda: self.centerRequested.emit(True, True))
        ml.addWidget(self.btn_center_both)

        hint = QLabel("Magnetic snap enabled (hold Shift for free move)")
        hint.setStyleSheet("color: #9e9e9e; margin-left: 8px;")
        ml.addWidget(hint)
        ml.addStretch(1)
        self._add_page("move", move_w)

        # 7. Hand / Nav page
        hand_w = QWidget()
        hl = QHBoxLayout(hand_w)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(6)
        btn_fit = QPushButton("Fit to View")
        btn_fit.clicked.connect(self.fitViewRequested.emit)
        hl.addWidget(btn_fit)
        btn_100 = QPushButton("100%")
        btn_100.clicked.connect(self.zoom100Requested.emit)
        hl.addWidget(btn_100)
        hand_hint = QLabel("Drag to pan viewport (Space+Drag or Middle-Drag from any tool)")
        hand_hint.setStyleSheet("color: #9e9e9e; margin-left: 6px;")
        hl.addWidget(hand_hint)
        hl.addStretch(1)
        self._add_page("hand", hand_w)

        # 8. Selection page (Marquee / Lasso)
        sel_w = QWidget()
        sel_l = QHBoxLayout(sel_w)
        sel_l.setContentsMargins(0, 0, 0, 0)
        sel_l.setSpacing(8)
        btn_clear_sel = QPushButton("Clear Selection")
        btn_clear_sel.clicked.connect(self.clearSelectionRequested.emit)
        sel_l.addWidget(btn_clear_sel)
        sel_hint = QLabel("Drag to select (Shift: Add, Alt: Subtract, Click: Deselect)")
        sel_hint.setStyleSheet("color: #9e9e9e;")
        sel_l.addWidget(sel_hint)
        sel_l.addStretch(1)
        self._add_page("selection", sel_w)

        # 9. Crop / Transform action page
        act_w = QWidget()
        al = QHBoxLayout(act_w)
        al.setContentsMargins(0, 0, 0, 0)
        al.setSpacing(6)
        self.btn_tool_apply = QPushButton("Apply (Enter)")
        self.btn_tool_apply.clicked.connect(lambda: self.toolActionRequested.emit("apply"))
        al.addWidget(self.btn_tool_apply)
        self.btn_tool_cancel = QPushButton("Cancel (Esc)")
        self.btn_tool_cancel.clicked.connect(lambda: self.toolActionRequested.emit("cancel"))
        al.addWidget(self.btn_tool_cancel)
        self.action_hint_label = QLabel("")
        self.action_hint_label.setStyleSheet("color: #9e9e9e; margin-left: 6px;")
        al.addWidget(self.action_hint_label)
        al.addStretch(1)
        self._add_page("action", act_w)

        # 10. Info fallback page (Eyedropper / Gradient)
        info_w = QWidget()
        il = QHBoxLayout(info_w)
        il.setContentsMargins(0, 0, 0, 0)
        self.info_hint_label = QLabel("")
        self.info_hint_label.setStyleSheet("color: #b0b0b0;")
        il.addWidget(self.info_hint_label)
        il.addStretch(1)
        self._add_page("info", info_w)

    def set_active_tool(self, tool_id: str, display_name: Optional[str] = None) -> None:
        self._current_tool_id = str(tool_id)
        label_map = {
            "move": "Move",
            "hand": "Hand",
            "brush": "Brush",
            "eraser": "Eraser",
            "smudge": "Smudge",
            "blur_sharpen": "Blur / Sharpen",
            "marquee": "Rect Marquee",
            "lasso": "Lasso",
            "magic_wand": "Magic Wand",
            "crop": "Crop",
            "transform": "Free Transform",
            "eyedropper": "Eyedropper",
            "bucket_fill": "Bucket Fill",
            "gradient": "Gradient",
            "text": "Text",
        }
        self.tool_name_label.setText(display_name or label_map.get(tool_id, tool_id.title()))
        self.sync_from_context()

        if tool_id in ("brush", "eraser"):
            self.edit_mask_chk.setVisible(True)
            self.stack.setCurrentIndex(self._page_indices["brush"])
        elif tool_id == "blur_sharpen":
            self.stack.setCurrentIndex(self._page_indices["blur_sharpen"])
        elif tool_id == "smudge":
            self.stack.setCurrentIndex(self._page_indices["smudge"])
        elif tool_id in ("magic_wand", "bucket_fill"):
            self.tol_clear_sel_btn.setVisible(tool_id == "magic_wand")
            self.stack.setCurrentIndex(self._page_indices["tolerance"])
        elif tool_id == "text":
            self.stack.setCurrentIndex(self._page_indices["text"])
        elif tool_id == "move":
            self.stack.setCurrentIndex(self._page_indices["move"])
        elif tool_id == "hand":
            self.stack.setCurrentIndex(self._page_indices["hand"])
        elif tool_id in ("marquee", "lasso"):
            self.stack.setCurrentIndex(self._page_indices["selection"])
        elif tool_id == "crop":
            self.action_hint_label.setText("Drag crop box — press Enter to crop or Esc to cancel")
            self.stack.setCurrentIndex(self._page_indices["action"])
        elif tool_id == "transform":
            self.action_hint_label.setText("Drag handles to scale/rotate (Shift: lock aspect) — Enter to commit")
            self.stack.setCurrentIndex(self._page_indices["action"])
        elif tool_id == "gradient":
            self.info_hint_label.setText("Drag on canvas to draw a linear gradient from FG to BG color")
            self.stack.setCurrentIndex(self._page_indices["info"])
        elif tool_id == "eyedropper":
            self.info_hint_label.setText("Click on canvas to sample Foreground color (also Alt+Click with Brush/Fill)")
            self.stack.setCurrentIndex(self._page_indices["info"])
        else:
            self.info_hint_label.setText("")
            self.stack.setCurrentIndex(self._page_indices["info"])

    def sync_from_context(self) -> None:
        self._syncing = True
        try:
            ctx = self.context
            self.brush_radius_spin.setValue(float(ctx.brush_radius))
            h_pct = int(round(ctx.brush_hardness * 100))
            o_pct = int(round(ctx.brush_opacity * 100))
            f_pct = int(round(ctx.brush_flow * 100))
            self.brush_hardness_slider.setValue(h_pct)
            self.brush_hardness_val.setText(f"{h_pct}%")
            self.brush_opacity_slider.setValue(o_pct)
            self.brush_opacity_val.setText(f"{o_pct}%")
            self.brush_flow_slider.setValue(f_pct)
            self.brush_flow_val.setText(f"{f_pct}%")
            self.edit_mask_chk.setChecked(bool(ctx.edit_mask))

            mode_idx = max(0, self.filter_mode_combo.findData(int(ctx.filter_mode)))
            self.filter_mode_combo.setCurrentIndex(mode_idx)
            k_idx = max(0, self.filter_kernel_combo.findData(int(ctx.filter_kernel_size)))
            self.filter_kernel_combo.setCurrentIndex(k_idx)
            self.filter_radius_spin.setValue(float(ctx.brush_radius))
            fs_pct = int(round(ctx.filter_strength * 100))
            self.filter_strength_slider.setValue(fs_pct)
            self.filter_strength_val.setText(f"{fs_pct}%")

            self.smudge_radius_spin.setValue(float(ctx.brush_radius))
            ss_pct = int(round(ctx.smudge_strength * 100))
            self.smudge_strength_slider.setValue(ss_pct)
            self.smudge_strength_val.setText(f"{ss_pct}%")

            if self._current_tool_id == "magic_wand":
                self.tolerance_spin.setValue(int(ctx.wand_tolerance))
                self.contiguous_chk.setChecked(bool(ctx.wand_contiguous))
            else:
                self.tolerance_spin.setValue(int(ctx.fill_tolerance))
                self.contiguous_chk.setChecked(bool(ctx.fill_contiguous))

            self.font_combo.setCurrentText(str(ctx.font_family))
            self.font_size_spin.setValue(int(ctx.font_size))
            self.font_bold_chk.setChecked(bool(ctx.font_bold))
            if self.text_content_edit.text() != str(ctx.text_content):
                self.text_content_edit.setText(str(ctx.text_content))
        finally:
            self._syncing = False

    def _on_brush_changed(self) -> None:
        if self._syncing:
            return
        self.context.brush_radius = float(self.brush_radius_spin.value())
        h_pct = self.brush_hardness_slider.value()
        o_pct = self.brush_opacity_slider.value()
        f_pct = self.brush_flow_slider.value()
        self.context.brush_hardness = h_pct / 100.0
        self.context.brush_opacity = o_pct / 100.0
        self.context.brush_flow = f_pct / 100.0
        self.context.edit_mask = self.edit_mask_chk.isChecked()
        self.brush_hardness_val.setText(f"{h_pct}%")
        self.brush_opacity_val.setText(f"{o_pct}%")
        self.brush_flow_val.setText(f"{f_pct}%")
        self.optionsChanged.emit()

    def _on_filter_changed(self) -> None:
        if self._syncing:
            return
        self.context.filter_mode = int(self.filter_mode_combo.currentData() or 0)
        self.context.filter_kernel_size = int(self.filter_kernel_combo.currentData() or 3)
        self.context.brush_radius = float(self.filter_radius_spin.value())
        fs_pct = self.filter_strength_slider.value()
        self.context.filter_strength = fs_pct / 100.0
        self.filter_strength_val.setText(f"{fs_pct}%")
        self.optionsChanged.emit()

    def _on_smudge_changed(self) -> None:
        if self._syncing:
            return
        self.context.brush_radius = float(self.smudge_radius_spin.value())
        ss_pct = self.smudge_strength_slider.value()
        self.context.smudge_strength = ss_pct / 100.0
        self.smudge_strength_val.setText(f"{ss_pct}%")
        self.optionsChanged.emit()

    def _on_tolerance_changed(self) -> None:
        if self._syncing:
            return
        tol = int(self.tolerance_spin.value())
        cont = self.contiguous_chk.isChecked()
        if self._current_tool_id == "magic_wand":
            self.context.wand_tolerance = tol
            self.context.wand_contiguous = cont
        else:
            self.context.fill_tolerance = tol
            self.context.fill_contiguous = cont
        self.optionsChanged.emit()

    def _on_text_changed(self) -> None:
        if self._syncing:
            return
        self.context.font_family = self.font_combo.currentText() or "Segoe UI"
        self.context.font_size = int(self.font_size_spin.value())
        self.context.font_bold = self.font_bold_chk.isChecked()
        self.context.text_content = self.text_content_edit.text()
        self.optionsChanged.emit()


class ColorPanel(QWidget):
    """Foreground/Background color swatches, swap/reset buttons, and quick palette."""

    fgColorChanged = Signal(tuple)
    bgColorChanged = Signal(tuple)

    def __init__(self, tool_context: ToolContext, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.context = tool_context

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # FG / BG header row
        top_row = QHBoxLayout()
        top_row.setSpacing(8)

        self.fg_button = QPushButton()
        self.fg_button.setFixedSize(40, 40)
        self.fg_button.setToolTip("Foreground Color (Click to pick)")
        self.fg_button.clicked.connect(self._pick_fg_color)

        self.bg_button = QPushButton()
        self.bg_button.setFixedSize(34, 34)
        self.bg_button.setToolTip("Background Color (Click to pick)")
        self.bg_button.clicked.connect(self._pick_bg_color)

        ctrl_col = QVBoxLayout()
        ctrl_col.setSpacing(4)
        self.btn_swap = QPushButton("Swap FG/BG")
        self.btn_swap.setToolTip("Swap Foreground and Background Colors (X)")
        self.btn_swap.clicked.connect(self.swap_colors)
        self.btn_reset = QPushButton("Reset B/W")
        self.btn_reset.setToolTip("Reset Foreground=White / Background=Black (D)")
        self.btn_reset.clicked.connect(self.reset_colors)
        ctrl_col.addWidget(self.btn_swap)
        ctrl_col.addWidget(self.btn_reset)

        top_row.addWidget(self.fg_button)
        top_row.addWidget(self.bg_button)
        top_row.addLayout(ctrl_col, 1)
        layout.addLayout(top_row)

        self.color_readout = QLabel("")
        self.color_readout.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        layout.addWidget(self.color_readout)

        # Quick Swatch Palette Grid (2 rows x 8 columns)
        palette_grid = QGridLayout()
        palette_grid.setSpacing(4)
        self.swatch_buttons: list[QPushButton] = []
        for idx, (r, g, b) in enumerate(_PALETTE_SWATCHES):
            btn = QPushButton()
            btn.setFixedSize(22, 22)
            btn.setToolTip(f"RGB({r}, {g}, {b}) — Click to set FG")
            btn.setStyleSheet(
                f"background-color: rgb({r}, {g}, {b}); border: 1px solid #4a4a50; border-radius: 3px; padding: 0px;"
            )
            btn.clicked.connect(lambda _chk=False, col=(r, g, b, 255): self.set_fg_color(col))
            palette_grid.addWidget(btn, idx // 8, idx % 8)
            self.swatch_buttons.append(btn)

        layout.addLayout(palette_grid)
        layout.addStretch(1)
        self.sync_from_context()

    def _update_swatch_styles(self) -> None:
        fr, fg, fb, fa = self.context.fg_color
        br, bg, bb, ba = self.context.bg_color
        self.fg_button.setStyleSheet(
            f"background-color: rgba({fr}, {fg}, {fb}, {fa}); border: 2px solid #ffffff; border-radius: 4px;"
        )
        self.bg_button.setStyleSheet(
            f"background-color: rgba({br}, {bg}, {bb}, {ba}); border: 2px solid #888888; border-radius: 4px;"
        )
        self.color_readout.setText(
            f"FG: #{fr:02X}{fg:02X}{fb:02X} ({fa})   BG: #{br:02X}{bg:02X}{bb:02X}"
        )

    def sync_from_context(self) -> None:
        self._update_swatch_styles()

    def set_fg_color(self, rgba: tuple[int, int, int, int]) -> None:
        norm = tuple(max(0, min(255, int(c))) for c in rgba)
        self.context.fg_color = (norm[0], norm[1], norm[2], norm[3])
        self._update_swatch_styles()
        self.fgColorChanged.emit(self.context.fg_color)

    def set_bg_color(self, rgba: tuple[int, int, int, int]) -> None:
        norm = tuple(max(0, min(255, int(c))) for c in rgba)
        self.context.bg_color = (norm[0], norm[1], norm[2], norm[3])
        self._update_swatch_styles()
        self.bgColorChanged.emit(self.context.bg_color)

    def swap_colors(self) -> None:
        fg, bg = self.context.fg_color, self.context.bg_color
        self.context.fg_color = bg
        self.context.bg_color = fg
        self._update_swatch_styles()
        self.fgColorChanged.emit(self.context.fg_color)
        self.bgColorChanged.emit(self.context.bg_color)

    def reset_colors(self) -> None:
        self.context.fg_color = (255, 255, 255, 255)
        self.context.bg_color = (0, 0, 0, 255)
        self._update_swatch_styles()
        self.fgColorChanged.emit(self.context.fg_color)
        self.bgColorChanged.emit(self.context.bg_color)

    def _pick_fg_color(self) -> None:
        color = QColorDialog.getColor(
            _rgba_to_qcolor(self.context.fg_color),
            self,
            "Select Foreground Color",
            QColorDialog.ShowAlphaChannel,
        )
        if color.isValid():
            self.set_fg_color(_qcolor_to_rgba(color))

    def _pick_bg_color(self) -> None:
        color = QColorDialog.getColor(
            _rgba_to_qcolor(self.context.bg_color),
            self,
            "Select Background Color",
            QColorDialog.ShowAlphaChannel,
        )
        if color.isValid():
            self.set_bg_color(_qcolor_to_rgba(color))


class _LayerRowWidget(QWidget):
    """Row widget rendered inside LayersPanel's QListWidget."""

    visibilityToggled = Signal(int)

    def __init__(self, layer: Layer, layer_index: int, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.layer_index = layer_index

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(6)

        self.vis_btn = QToolButton()
        self.vis_btn.setFixedSize(22, 22)
        self.vis_btn.setText("👁" if layer.visible else "—")
        self.vis_btn.setToolTip("Toggle Layer Visibility")
        self.vis_btn.clicked.connect(lambda: self.visibilityToggled.emit(self.layer_index))
        layout.addWidget(self.vis_btn)

        if layer.clipping_mask:
            clip_lbl = QLabel("↳")
            clip_lbl.setStyleSheet("color: #38bdf8; font-weight: bold;")
            clip_lbl.setToolTip("Clipped to base layer below")
            layout.addWidget(clip_lbl)

        name_lbl = QLabel(layer.name)
        name_lbl.setStyleSheet(
            "color: #e5e5e5;" if layer.visible else "color: #777777; font-style: italic;"
        )
        layout.addWidget(name_lbl, 1)

        if layer.mask is not None:
            mask_lbl = QLabel("[M]" if layer.mask.enabled else "[M×]")
            mask_lbl.setStyleSheet(
                "color: #a3e635; font-size: 10px; font-weight: 600;"
                if layer.mask.enabled
                else "color: #888888; font-size: 10px;"
            )
            mask_lbl.setToolTip("Layer Mask attached")
            layout.addWidget(mask_lbl)

        if layer.locked:
            lock_lbl = QLabel("🔒")
            lock_lbl.setToolTip("Layer Locked")
            layout.addWidget(lock_lbl)


class LayersPanel(QWidget):
    """Photoshop-style layer stack panel with blend modes, opacity, masks, and clipping."""

    layersChanged = Signal()
    activeLayerChanged = Signal(int)

    def __init__(
        self,
        tool_context: Optional[ToolContext] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.context = tool_context
        self._document: Optional[DocumentSession] = None
        self._syncing: bool = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Blend mode & Opacity controls
        top_row = QHBoxLayout()
        top_row.setSpacing(6)

        self.blend_combo = QComboBox()
        for mode in (
            BlendMode.NORMAL,
            BlendMode.MULTIPLY,
            BlendMode.SCREEN,
            BlendMode.OVERLAY,
            BlendMode.ADD,
        ):
            self.blend_combo.addItem(BLEND_MODE_LABELS[mode], mode)
        self.blend_combo.currentIndexChanged.connect(self._on_blend_changed)
        top_row.addWidget(self.blend_combo, 1)

        top_row.addWidget(QLabel("Opacity:"))
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setFixedWidth(75)
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        top_row.addWidget(self.opacity_slider)

        self.opacity_label = QLabel("100%")
        self.opacity_label.setFixedWidth(36)
        top_row.addWidget(self.opacity_label)
        layout.addLayout(top_row)

        # Layer stack list (top-to-bottom)
        self.layer_list = QListWidget()
        self.layer_list.currentRowChanged.connect(self._on_row_selected)
        self.layer_list.itemDoubleClicked.connect(self._on_rename_requested)
        layout.addWidget(self.layer_list, 1)

        # Mask & Attribute toggles row
        attr_row = QHBoxLayout()
        attr_row.setSpacing(4)

        self.btn_visibility = QPushButton("👁 Vis")
        self.btn_visibility.setToolTip("Toggle Active Layer Visibility")
        self.btn_visibility.clicked.connect(self.toggle_visibility)
        attr_row.addWidget(self.btn_visibility)

        self.btn_lock = QPushButton("🔒 Lock")
        self.btn_lock.setToolTip("Toggle Active Layer Lock")
        self.btn_lock.clicked.connect(self.toggle_lock)
        attr_row.addWidget(self.btn_lock)

        self.btn_clipping = QPushButton("↳ Clip")
        self.btn_clipping.setToolTip("Toggle Clipping Mask to Layer Below")
        self.btn_clipping.clicked.connect(self.toggle_clipping_mask)
        attr_row.addWidget(self.btn_clipping)

        self.btn_mask = QPushButton("Mask")
        self.btn_mask.setToolTip("Add or Delete Layer Mask")
        self.btn_mask.clicked.connect(self.add_or_delete_mask)
        attr_row.addWidget(self.btn_mask)

        self.btn_toggle_mask = QPushButton("M On/Off")
        self.btn_toggle_mask.setToolTip("Enable / Disable Layer Mask")
        self.btn_toggle_mask.clicked.connect(self.toggle_mask_enabled)
        attr_row.addWidget(self.btn_toggle_mask)
        layout.addLayout(attr_row)

        # Paint-mask checkbox (if ToolContext is wired)
        mask_mode_row = QHBoxLayout()
        mask_mode_row.setSpacing(6)
        self.chk_edit_mask = QCheckBox("Paint on Layer Mask")
        self.chk_edit_mask.toggled.connect(self._on_edit_mask_toggled)
        mask_mode_row.addWidget(self.chk_edit_mask)
        mask_mode_row.addStretch(1)
        layout.addLayout(mask_mode_row)

        # Bottom CRUD / Reorder buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)

        self.btn_add = QPushButton("+ Add")
        self.btn_add.setToolTip("Add New Layer")
        self.btn_add.clicked.connect(self.add_layer)
        btn_row.addWidget(self.btn_add)

        self.btn_duplicate = QPushButton("Dup")
        self.btn_duplicate.setToolTip("Duplicate Active Layer")
        self.btn_duplicate.clicked.connect(self.duplicate_layer)
        btn_row.addWidget(self.btn_duplicate)

        self.btn_up = QPushButton("↑")
        self.btn_up.setToolTip("Move Layer Up")
        self.btn_up.clicked.connect(self.move_layer_up)
        btn_row.addWidget(self.btn_up)

        self.btn_down = QPushButton("↓")
        self.btn_down.setToolTip("Move Layer Down")
        self.btn_down.clicked.connect(self.move_layer_down)
        btn_row.addWidget(self.btn_down)

        self.btn_merge_down = QPushButton("Merge")
        self.btn_merge_down.setToolTip("Merge Down")
        self.btn_merge_down.clicked.connect(self.merge_down)
        btn_row.addWidget(self.btn_merge_down)

        self.btn_delete = QPushButton("Del")
        self.btn_delete.setToolTip("Delete Active Layer")
        self.btn_delete.clicked.connect(self.delete_layer)
        btn_row.addWidget(self.btn_delete)

        layout.addLayout(btn_row)
        self.refresh()

    @property
    def document(self) -> Optional[DocumentSession]:
        return self._document

    def set_document(self, doc: Optional[DocumentSession]) -> None:
        self._document = doc
        self.refresh()

    def _row_to_layer_index(self, row: int) -> int:
        if self._document is None or row < 0:
            return -1
        return len(self._document.layers) - 1 - row

    def _layer_index_to_row(self, index: int) -> int:
        if self._document is None or index < 0:
            return -1
        return len(self._document.layers) - 1 - index

    def refresh(self) -> None:
        self._syncing = True
        try:
            self.layer_list.clear()
            doc = self._document
            if doc is None or not doc.layers:
                self.blend_combo.setEnabled(False)
                self.opacity_slider.setEnabled(False)
                self.opacity_label.setText("0%")
                return

            self.blend_combo.setEnabled(True)
            self.opacity_slider.setEnabled(True)

            # Display top-to-bottom (highest index at row 0)
            for idx in range(len(doc.layers) - 1, -1, -1):
                layer = doc.layers[idx]
                item = QListWidgetItem()
                item.setData(Qt.UserRole, idx)
                item.setSizeHint(QSize(0, 30))
                self.layer_list.addItem(item)

                row_widget = _LayerRowWidget(layer, idx)
                row_widget.visibilityToggled.connect(self.toggle_visibility)
                self.layer_list.setItemWidget(item, row_widget)

            active_row = self._layer_index_to_row(doc.active_layer_index)
            if 0 <= active_row < self.layer_list.count():
                self.layer_list.setCurrentRow(active_row)

            layer = doc.active_layer
            if layer is not None:
                b_idx = max(0, self.blend_combo.findData(layer.blend_mode))
                self.blend_combo.setCurrentIndex(b_idx)
                pct = int(round(layer.opacity * 100))
                self.opacity_slider.setValue(pct)
                self.opacity_label.setText(f"{pct}%")

                self.btn_mask.setText("- Mask" if layer.mask is not None else "+ Mask")
                self.btn_toggle_mask.setEnabled(layer.mask is not None)
                self.chk_edit_mask.setEnabled(layer.mask is not None)
                if self.context is not None:
                    self.chk_edit_mask.setChecked(
                        bool(self.context.edit_mask and layer.mask is not None)
                    )
                self.btn_delete.setEnabled(len(doc.layers) > 1)
                self.btn_merge_down.setEnabled(doc.active_layer_index > 0)
                self.btn_up.setEnabled(doc.active_layer_index < len(doc.layers) - 1)
                self.btn_down.setEnabled(doc.active_layer_index > 0)
        finally:
            self._syncing = False

    def _emit_change(self) -> None:
        self.refresh()
        self.layersChanged.emit()

    def _on_row_selected(self, row: int) -> None:
        if self._syncing or self._document is None:
            return
        idx = self._row_to_layer_index(row)
        if 0 <= idx < len(self._document.layers):
            self._document.set_active_layer(idx)
            self.refresh()
            self.activeLayerChanged.emit(idx)

    def _on_blend_changed(self) -> None:
        if self._syncing or self._document is None:
            return
        mode = self.blend_combo.currentData()
        if mode is not None and self._document.active_layer is not None:
            self._document.set_layer_blend_mode(self._document.active_layer_index, mode)
            self._emit_change()

    def _on_opacity_changed(self, value: int) -> None:
        self.opacity_label.setText(f"{value}%")
        if self._syncing or self._document is None or self._document.active_layer is None:
            return
        self._document.set_layer_opacity(self._document.active_layer_index, value / 100.0)
        self.layersChanged.emit()

    def _on_edit_mask_toggled(self, checked: bool) -> None:
        if self._syncing or self.context is None:
            return
        self.context.edit_mask = bool(checked)
        self.layersChanged.emit()

    def _on_rename_requested(self, item: QListWidgetItem) -> None:
        if self._document is None:
            return
        idx = int(item.data(Qt.UserRole))
        if not (0 <= idx < len(self._document.layers)):
            return
        old_name = self._document.layers[idx].name
        new_name, ok = QInputDialog.getText(self, "Rename Layer", "Layer Name:", text=old_name)
        if ok and new_name.strip():
            self._document.rename_layer(idx, new_name.strip())
            self._emit_change()

    def add_layer(self) -> Optional[Layer]:
        if self._document is None:
            return None
        layer = self._document.add_layer(index=self._document.active_layer_index + 1)
        self._emit_change()
        return layer

    def delete_layer(self) -> None:
        if self._document is None or len(self._document.layers) <= 1:
            return
        self._document.remove_layer(self._document.active_layer_index)
        self._emit_change()

    def duplicate_layer(self) -> Optional[Layer]:
        if self._document is None or self._document.active_layer is None:
            return None
        layer = self._document.duplicate_layer(self._document.active_layer_index)
        self._emit_change()
        return layer

    def move_layer_up(self) -> None:
        if self._document is None:
            return
        idx = self._document.active_layer_index
        if idx < len(self._document.layers) - 1:
            self._document.move_layer(idx, idx + 1)
            self._emit_change()

    def move_layer_down(self) -> None:
        if self._document is None:
            return
        idx = self._document.active_layer_index
        if idx > 0:
            self._document.move_layer(idx, idx - 1)
            self._emit_change()

    def merge_down(self) -> None:
        if self._document is None:
            return
        if self._document.active_layer_index > 0:
            self._document.merge_down(self._document.active_layer_index)
            self._emit_change()

    def toggle_visibility(self, index: Optional[int] = None) -> None:
        if self._document is None:
            return
        idx = self._document.active_layer_index if index is None or isinstance(index, bool) else int(index)
        if 0 <= idx < len(self._document.layers):
            layer = self._document.layers[idx]
            self._document.set_layer_visible(idx, not layer.visible)
            self._emit_change()

    def toggle_lock(self) -> None:
        if self._document is None or self._document.active_layer is None:
            return
        idx = self._document.active_layer_index
        self._document.set_layer_locked(idx, not self._document.active_layer.locked)
        self._emit_change()

    def toggle_clipping_mask(self) -> None:
        if self._document is None or self._document.active_layer is None:
            return
        idx = self._document.active_layer_index
        self._document.set_layer_clipping_mask(idx, not self._document.active_layer.clipping_mask)
        self._emit_change()

    def add_or_delete_mask(self) -> None:
        if self._document is None or self._document.active_layer is None:
            return
        doc = self._document
        layer = doc.active_layer

        if layer.mask is None:
            new_mask = LayerMask(layer.width, layer.height, fill_value=255)

            def do(d: DocumentSession) -> None:
                layer.mask = new_mask

            def undo(d: DocumentSession) -> None:
                layer.mask = None
                if self.context is not None:
                    self.context.edit_mask = False

            do(doc)
            doc.undo_stack.push(
                CallableCommand(
                    undo_fn=undo,
                    redo_fn=do,
                    description=f"Add Mask '{layer.name}'",
                    estimated_bytes=int(new_mask.buffer.nbytes),
                )
            )
            doc.modified = True
            doc.invalidate_composite()
        else:
            old_mask = layer.mask

            def do(d: DocumentSession) -> None:
                layer.mask = None
                if self.context is not None:
                    self.context.edit_mask = False

            def undo(d: DocumentSession) -> None:
                layer.mask = old_mask

            do(doc)
            doc.undo_stack.push(
                CallableCommand(
                    undo_fn=undo,
                    redo_fn=do,
                    description=f"Delete Mask '{layer.name}'",
                    estimated_bytes=int(old_mask.buffer.nbytes),
                )
            )
            doc.modified = True
            doc.invalidate_composite()

        self._emit_change()

    def toggle_mask_enabled(self) -> None:
        if self._document is None or self._document.active_layer is None:
            return
        layer = self._document.active_layer
        if layer.mask is None:
            return
        doc = self._document
        mask = layer.mask
        old_val = mask.enabled
        new_val = not old_val

        def do(d: DocumentSession) -> None:
            mask.enabled = new_val

        def undo(d: DocumentSession) -> None:
            mask.enabled = old_val

        do(doc)
        doc.undo_stack.push(
            CallableCommand(undo_fn=undo, redo_fn=do, description="Toggle Layer Mask")
        )
        doc.modified = True
        doc.invalidate_composite()
        self._emit_change()
