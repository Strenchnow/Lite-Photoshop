"""Main tabbed workspace window integrating CanvasView, panels, tools, menus, dialogs, and Zen mode."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QAction, QActionGroup, QGuiApplication, QImage, QKeySequence
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDockWidget,
    QFileDialog,
    QFormLayout,
    QKeySequenceEdit,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QSpinBox,
    QTabBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from app.core.ai_bg_remove import apply_ai_background_removal
from app.core.document import DocumentSession, DocumentTabManager
from app.core.layer import Layer
from app.tools import BaseTool, ToolContext, ToolType, create_default_tools
from app.ui.canvas_view import CanvasView
from app.ui.dialogs import (
    BatchProcessorDialog,
    ChromaKeyDialog,
    GridSplitterDialog,
    HotkeyEditorDialog,
    QuickExportDialog,
    SmartAlphaSplitterDialog,
    load_keybindings,
)
from app.ui.panels import ColorPanel, LayersPanel, ToolOptionsBar


_TOOL_SPECS: tuple[tuple[str, str, str], ...] = (
    (ToolType.MOVE.value, "Move", "Move Layer (V)"),
    (ToolType.HAND.value, "Hand", "Hand Pan (H)"),
    (ToolType.MARQUEE.value, "Marquee", "Rectangular Marquee (M)"),
    (ToolType.LASSO.value, "Lasso", "Freehand Lasso (L)"),
    (ToolType.MAGIC_WAND.value, "Wand", "Magic Wand (W)"),
    (ToolType.CROP.value, "Crop", "Crop Canvas (C)"),
    (ToolType.TRANSFORM.value, "Transform", "Free Transform (Ctrl+T)"),
    (ToolType.BRUSH.value, "Brush", "Brush (B)"),
    (ToolType.ERASER.value, "Eraser", "Eraser (E)"),
    (ToolType.SMUDGE.value, "Smudge", "Smudge (S)"),
    (ToolType.BLUR_SHARPEN.value, "Blur/Sharp", "Local Blur / Sharpen (R)"),
    (ToolType.BUCKET_FILL.value, "Fill", "Bucket Fill (G)"),
    (ToolType.GRADIENT.value, "Gradient", "Linear Gradient (K)"),
    (ToolType.EYEDROPPER.value, "Picker", "Eyedropper (I)"),
    (ToolType.TEXT.value, "Text", "Text Tool (T)"),
)


def qimage_to_rgba_numpy(qimg: QImage) -> np.ndarray:
    """Convert a non-null QImage into a contiguous (H, W, 4) uint8 RGBA array."""
    if qimg.isNull():
        raise ValueError("Cannot convert null QImage to RGBA array")
    rgba_img = qimg.convertToFormat(QImage.Format_RGBA8888)
    w, h = rgba_img.width(), rgba_img.height()
    bpl = rgba_img.bytesPerLine()
    ptr = rgba_img.constBits()
    arr = np.frombuffer(ptr, dtype=np.uint8, count=h * bpl).reshape((h, bpl))
    return np.ascontiguousarray(arr[:, : w * 4].reshape((h, w, 4)).copy(), dtype=np.uint8)


class _NewCanvasDialog(QDialog):
    """Lightweight dialog for custom document dimensions."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("New Canvas")
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.width_spin = QSpinBox()
        self.width_spin.setRange(1, 16384)
        self.width_spin.setValue(1024)
        self.height_spin = QSpinBox()
        self.height_spin.setRange(1, 16384)
        self.height_spin.setValue(1024)
        self.bg_combo = QComboBox()
        self.bg_combo.addItems(["Transparent", "White", "Black"])
        form.addRow("Width (px):", self.width_spin)
        form.addRow("Height (px):", self.height_spin)
        form.addRow("Background:", self.bg_combo)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class MainWindow(QMainWindow):
    """Main LitePhotoEditor workspace window."""

    def __init__(
        self,
        tab_manager: Optional[DocumentTabManager] = None,
        create_initial_doc: bool = True,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("LitePhotoEditor")
        self.resize(1280, 820)

        self.tab_manager = tab_manager if tab_manager is not None else DocumentTabManager()
        self.tool_context = ToolContext(
            on_color_picked=self._on_color_picked,
            on_canvas_invalidated=self._refresh_canvas_and_tabs,
            on_document_resized=self._on_document_resized,
        )
        self.tools: dict[str, BaseTool] = create_default_tools(self.tool_context)
        self.active_tool_id: str = ToolType.BRUSH.value

        self._internal_clipboard_rgba: Optional[np.ndarray] = None
        self.is_zen_mode: bool = False
        self._pre_zen_visibility: dict[str, bool] = {}

        self.tool_actions: dict[str, QAction] = {}
        self.editor_actions: dict[str, QAction] = {}
        self.keybindings: dict[str, str] = {}

        self._build_central_area()
        self._build_docks()
        self._build_tools_toolbar()
        self._build_menus()
        self._build_status_bar()
        self._connect_signals()

        self.keybindings = load_keybindings()
        self.apply_keybindings(self.keybindings)

        if create_initial_doc and not self.tab_manager.sessions:
            self.new_document(800, 600, title="Untitled-1")
        else:
            self._sync_tabs()

        self.select_tool(ToolType.BRUSH.value)
        self.canvas_view.installEventFilter(self)

    def _build_central_area(self) -> None:
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.tab_bar = QTabBar()
        self.tab_bar.setTabsClosable(True)
        self.tab_bar.setMovable(False)
        self.tab_bar.setExpanding(False)
        self.tab_bar.setDocumentMode(True)
        layout.addWidget(self.tab_bar)

        self.tool_options_bar = ToolOptionsBar(self.tool_context)
        layout.addWidget(self.tool_options_bar)

        self.canvas_view = CanvasView(tool_context=self.tool_context)
        self.canvas_view.set_hand_tool(self.tools.get(ToolType.HAND.value))
        layout.addWidget(self.canvas_view, 1)

        self.setCentralWidget(central)

    def _build_docks(self) -> None:
        self.color_panel = ColorPanel(self.tool_context)
        self.color_dock = QDockWidget("Colors & Swatches", self)
        self.color_dock.setObjectName("ColorDock")
        self.color_dock.setWidget(self.color_panel)
        self.color_dock.setMinimumWidth(240)
        self.addDockWidget(Qt.RightDockWidgetArea, self.color_dock)

        self.layers_panel = LayersPanel(self.tool_context)
        self.layers_dock = QDockWidget("Layers", self)
        self.layers_dock.setObjectName("LayersDock")
        self.layers_dock.setWidget(self.layers_panel)
        self.layers_dock.setMinimumWidth(240)
        self.addDockWidget(Qt.RightDockWidgetArea, self.layers_dock)

    def _build_tools_toolbar(self) -> None:
        self.tools_toolbar = QToolBar("Tools", self)
        self.tools_toolbar.setObjectName("ToolsToolBar")
        self.tools_toolbar.setMovable(False)
        self.tools_toolbar.setOrientation(Qt.Vertical)
        self.addToolBar(Qt.LeftToolBarArea, self.tools_toolbar)

        self._tool_action_group = QActionGroup(self)
        self._tool_action_group.setExclusive(True)

        for tool_id, label, tooltip in _TOOL_SPECS:
            act = QAction(label, self)
            act.setCheckable(True)
            act.setToolTip(tooltip)
            act.setData(tool_id)
            act.triggered.connect(lambda _chk=False, tid=tool_id: self.select_tool(tid))
            self._tool_action_group.addAction(act)
            self.tools_toolbar.addAction(act)
            self.tool_actions[tool_id] = act

    def _register_editor_action(
        self,
        key: str,
        text: str,
        slot,
        checkable: bool = False,
    ) -> QAction:
        act = QAction(text, self)
        act.setCheckable(checkable)
        act.triggered.connect(slot)
        self.addAction(act)
        self.editor_actions[key] = act
        return act

    def _build_menus(self) -> None:
        mb = self.menuBar()

        # File Menu
        file_menu = mb.addMenu("&File")
        file_menu.addAction(self._register_editor_action("new_tab", "&New Tab", self.new_document))
        act_new_custom = QAction("New &Custom Canvas...", self)
        act_new_custom.triggered.connect(self._open_new_custom_dialog)
        file_menu.addAction(act_new_custom)
        file_menu.addAction(self._register_editor_action("open_file", "&Open...", self.open_file))
        file_menu.addAction(self._register_editor_action("save_file", "&Save", self.save_file))
        file_menu.addSeparator()
        file_menu.addAction(
            self._register_editor_action("quick_export", "Quick &Export...", self.open_quick_export)
        )
        act_batch = QAction("&Batch Processor...", self)
        act_batch.triggered.connect(self.open_batch_processor)
        file_menu.addAction(act_batch)
        file_menu.addSeparator()
        act_close_tab = QAction("&Close Tab", self)
        act_close_tab.setShortcut(QKeySequence("Ctrl+W"))
        act_close_tab.triggered.connect(self.close_current_tab)
        file_menu.addAction(act_close_tab)
        act_exit = QAction("E&xit", self)
        act_exit.triggered.connect(self.close)
        file_menu.addAction(act_exit)

        # Edit Menu
        edit_menu = mb.addMenu("&Edit")
        edit_menu.addAction(self._register_editor_action("undo", "&Undo", self.undo))
        edit_menu.addAction(self._register_editor_action("redo", "&Redo", self.redo))
        edit_menu.addSeparator()
        act_copy = QAction("&Copy Selection", self)
        act_copy.setShortcut(QKeySequence("Ctrl+C"))
        act_copy.triggered.connect(self.copy_selection)
        edit_menu.addAction(act_copy)
        edit_menu.addAction(
            self._register_editor_action(
                "paste_clipboard", "&Paste / New Tab from Clipboard", self.paste_from_clipboard
            )
        )
        act_desel = QAction("&Deselect", self)
        act_desel.setShortcut(QKeySequence("Ctrl+D"))
        act_desel.triggered.connect(self.clear_selection)
        edit_menu.addAction(act_desel)
        edit_menu.addSeparator()
        act_hotkeys = QAction("&Keyboard Shortcuts...", self)
        act_hotkeys.triggered.connect(self.open_hotkey_editor)
        edit_menu.addAction(act_hotkeys)

        # Image Menu
        image_menu = mb.addMenu("&Image")
        act_trim = QAction("&Trim Transparent Pixels", self)
        act_trim.triggered.connect(self.trim_transparent)
        image_menu.addAction(act_trim)
        image_menu.addSeparator()
        act_ai_bg = QAction("&AI Remove Background", self)
        act_ai_bg.triggered.connect(self.run_ai_background_removal)
        image_menu.addAction(act_ai_bg)
        act_chroma = QAction("&Chroma Key...", self)
        act_chroma.triggered.connect(self.open_chroma_key)
        image_menu.addAction(act_chroma)
        image_menu.addSeparator()
        act_grid_split = QAction("&Grid Splitter...", self)
        act_grid_split.triggered.connect(self.open_grid_splitter)
        image_menu.addAction(act_grid_split)
        act_alpha_split = QAction("Smart &Alpha Splitter...", self)
        act_alpha_split.triggered.connect(self.open_smart_alpha_splitter)
        image_menu.addAction(act_alpha_split)

        # Layer Menu
        layer_menu = mb.addMenu("&Layer")
        act_add_l = QAction("&Add Layer", self)
        act_add_l.triggered.connect(self.layers_panel.add_layer)
        layer_menu.addAction(act_add_l)
        act_dup_l = QAction("&Duplicate Layer", self)
        act_dup_l.triggered.connect(self.layers_panel.duplicate_layer)
        layer_menu.addAction(act_dup_l)
        act_del_l = QAction("De&lete Layer", self)
        act_del_l.triggered.connect(self.layers_panel.delete_layer)
        layer_menu.addAction(act_del_l)
        act_merge_l = QAction("&Merge Down", self)
        act_merge_l.triggered.connect(self.layers_panel.merge_down)
        layer_menu.addAction(act_merge_l)
        layer_menu.addSeparator()
        act_mask = QAction("Add / Delete Layer &Mask", self)
        act_mask.triggered.connect(self.layers_panel.add_or_delete_mask)
        layer_menu.addAction(act_mask)
        act_clip = QAction("Toggle &Clipping Mask", self)
        act_clip.triggered.connect(self.layers_panel.toggle_clipping_mask)
        layer_menu.addAction(act_clip)
        layer_menu.addSeparator()
        act_ch = QAction("Center &Horizontal", self)
        act_ch.triggered.connect(lambda: self.center_active_layer(True, False))
        layer_menu.addAction(act_ch)
        act_cv = QAction("Center &Vertical", self)
        act_cv.triggered.connect(lambda: self.center_active_layer(False, True))
        layer_menu.addAction(act_cv)
        layer_menu.addAction(
            self._register_editor_action(
                "center_both", "Center &Both", lambda: self.center_active_layer(True, True)
            )
        )

        # Tools Menu
        tools_menu = mb.addMenu("&Tools")
        for tool_id, _, _ in _TOOL_SPECS:
            if tool_id in self.tool_actions:
                tools_menu.addAction(self.tool_actions[tool_id])
        tools_menu.addSeparator()
        act_swap = QAction("Swap FG/BG Colors", self)
        act_swap.setShortcut(QKeySequence("X"))
        act_swap.triggered.connect(self.color_panel.swap_colors)
        tools_menu.addAction(act_swap)
        act_reset_bw = QAction("Reset B/W Colors", self)
        act_reset_bw.setShortcut(QKeySequence("D"))
        act_reset_bw.triggered.connect(self.color_panel.reset_colors)
        tools_menu.addAction(act_reset_bw)

        # View Menu
        view_menu = mb.addMenu("&View")
        act_zoom_in = QAction("Zoom &In", self)
        act_zoom_in.setShortcut(QKeySequence("Ctrl+="))
        act_zoom_in.triggered.connect(lambda: self._zoom_by(1.25))
        view_menu.addAction(act_zoom_in)
        act_zoom_out = QAction("Zoom &Out", self)
        act_zoom_out.setShortcut(QKeySequence("Ctrl+-"))
        act_zoom_out.triggered.connect(lambda: self._zoom_by(0.8))
        view_menu.addAction(act_zoom_out)
        act_fit = QAction("&Fit to Screen", self)
        act_fit.setShortcut(QKeySequence("Ctrl+0"))
        act_fit.triggered.connect(self.canvas_view.fit_to_view)
        view_menu.addAction(act_fit)
        act_100 = QAction("&Actual Size (100%)", self)
        act_100.setShortcut(QKeySequence("Ctrl+1"))
        act_100.triggered.connect(self.canvas_view.set_zoom_100)
        view_menu.addAction(act_100)
        view_menu.addSeparator()
        view_menu.addAction(
            self._register_editor_action("zen_mode", "&Zen Mode", self.toggle_zen_mode, checkable=True)
        )
        view_menu.addSeparator()
        view_menu.addAction(self.layers_dock.toggleViewAction())
        view_menu.addAction(self.color_dock.toggleViewAction())

    def _build_status_bar(self) -> None:
        sb = self.statusBar()
        self.status_zoom_label = QLabel("Zoom: 100%")
        self.status_cursor_label = QLabel("X: 0, Y: 0")
        self.status_doc_label = QLabel("0 x 0 px")

        sb.addWidget(self.status_zoom_label)
        sb.addWidget(self.status_cursor_label)
        sb.addPermanentWidget(self.status_doc_label)

    def _connect_signals(self) -> None:
        self.tab_bar.currentChanged.connect(self._on_tab_changed)
        self.tab_bar.tabCloseRequested.connect(self.close_tab)

        self.canvas_view.cursorMoved.connect(self._on_cursor_moved)
        self.canvas_view.zoomChanged.connect(self._on_zoom_changed)
        self.canvas_view.documentModified.connect(self._on_canvas_document_modified)

        self.layers_panel.layersChanged.connect(self._on_layers_changed)
        self.layers_panel.activeLayerChanged.connect(self._on_active_layer_changed)

        self.tool_options_bar.optionsChanged.connect(self._on_tool_options_changed)
        self.tool_options_bar.centerRequested.connect(self.center_active_layer)
        self.tool_options_bar.clearSelectionRequested.connect(self.clear_selection)
        self.tool_options_bar.fitViewRequested.connect(self.canvas_view.fit_to_view)
        self.tool_options_bar.zoom100Requested.connect(self.canvas_view.set_zoom_100)
        self.tool_options_bar.toolActionRequested.connect(self._on_tool_action_requested)

        self.color_panel.fgColorChanged.connect(lambda _c: self.canvas_view.update())
        self.color_panel.bgColorChanged.connect(lambda _c: self.canvas_view.update())

    def apply_keybindings(self, bindings: dict[str, str]) -> None:
        self.keybindings = dict(bindings)
        for tool_id, act in self.tool_actions.items():
            seq = bindings.get(tool_id, "")
            act.setShortcut(QKeySequence.fromString(seq) if seq else QKeySequence())
            base_tip = act.text()
            act.setToolTip(f"{base_tip} ({seq})" if seq else base_tip)
        for key, act in self.editor_actions.items():
            seq = bindings.get(key, "")
            act.setShortcut(QKeySequence.fromString(seq) if seq else QKeySequence())

    @property
    def active_document(self) -> Optional[DocumentSession]:
        return self.tab_manager.active_document

    def select_tool(self, tool_id: str) -> None:
        tool = self.tools.get(tool_id)
        if tool is None:
            return
        self.active_tool_id = tool_id
        self.canvas_view.set_active_tool(tool)
        if tool_id in self.tool_actions:
            self.tool_actions[tool_id].setChecked(True)
        self.tool_options_bar.set_active_tool(tool_id, display_name=tool.name)
        self.layers_panel.refresh()

    # --- Tab & Document Lifecycle ---

    def _sync_tabs(self) -> None:
        self.tab_bar.blockSignals(True)
        try:
            while self.tab_bar.count() > len(self.tab_manager.sessions):
                self.tab_bar.removeTab(self.tab_bar.count() - 1)
            for idx, doc in enumerate(self.tab_manager.sessions):
                if idx < self.tab_bar.count():
                    self.tab_bar.setTabText(idx, doc.display_title)
                else:
                    self.tab_bar.addTab(doc.display_title)
            if 0 <= self.tab_manager.active_index < self.tab_bar.count():
                self.tab_bar.setCurrentIndex(self.tab_manager.active_index)
        finally:
            self.tab_bar.blockSignals(False)

        active = self.tab_manager.active_document
        self.canvas_view.set_document(active)
        self.layers_panel.set_document(active)
        self._update_status_bar()

    def _sync_tab_titles(self) -> None:
        for idx, doc in enumerate(self.tab_manager.sessions):
            if idx < self.tab_bar.count():
                title = doc.display_title
                if self.tab_bar.tabText(idx) != title:
                    self.tab_bar.setTabText(idx, title)

    def _on_tab_changed(self, index: int) -> None:
        if 0 <= index < len(self.tab_manager.sessions):
            self.switch_to_tab(index)

    def switch_to_tab(self, index: int) -> Optional[DocumentSession]:
        if not (0 <= index < len(self.tab_manager.sessions)):
            return None
        doc = self.tab_manager.set_active_tab(index)
        self._sync_tabs()
        return doc

    def new_document(
        self,
        width: int = 800,
        height: int = 600,
        title: Optional[str] = None,
        initial_fill: tuple[int, int, int, int] = (0, 0, 0, 0),
    ) -> DocumentSession:
        # Qt triggered(bool) passes False if connected directly without default check
        if isinstance(width, bool):
            width = 800
        doc = self.tab_manager.new_document(
            width=int(width),
            height=int(height),
            title=title,
            initial_fill=initial_fill,
            activate=True,
        )
        self._sync_tabs()
        if self.canvas_view.isVisible() and self.canvas_view.width() > 120 and self.canvas_view.height() > 120:
            self.canvas_view.fit_to_view()
        return doc

    def _open_new_custom_dialog(self) -> None:
        dlg = _NewCanvasDialog(self)
        if dlg.exec() == QDialog.Accepted:
            fill_map = {
                "Transparent": (0, 0, 0, 0),
                "White": (255, 255, 255, 255),
                "Black": (0, 0, 0, 255),
            }
            fill = fill_map.get(dlg.bg_combo.currentText(), (0, 0, 0, 0))
            self.new_document(dlg.width_spin.value(), dlg.height_spin.value(), initial_fill=fill)

    def open_file(self, file_path: Optional[Path] = None) -> Optional[DocumentSession]:
        if file_path is None or isinstance(file_path, bool):
            chosen, _ = QFileDialog.getOpenFileName(
                self,
                "Open Image",
                "",
                "Images (*.png *.jpg *.jpeg *.webp *.bmp);;All Files (*)",
            )
            if not chosen:
                return None
            target = Path(chosen)
        else:
            target = Path(file_path)

        with Image.open(target) as im:
            rgba = np.ascontiguousarray(np.asarray(im.convert("RGBA"), dtype=np.uint8))

        doc = self.tab_manager.new_document_from_rgba(rgba, title=target.name, activate=True)
        doc.file_path = target
        doc.modified = False
        self._sync_tabs()
        if self.canvas_view.isVisible() and self.canvas_view.width() > 120 and self.canvas_view.height() > 120:
            self.canvas_view.fit_to_view()
        return doc

    def save_file(self, file_path: Optional[Path] = None) -> Optional[Path]:
        doc = self.active_document
        if doc is None:
            return None

        if file_path is not None and not isinstance(file_path, bool):
            target = Path(file_path)
        elif doc.file_path is not None:
            target = doc.file_path
        else:
            chosen, _ = QFileDialog.getSaveFileName(
                self,
                "Save Image",
                f"{doc.title}.png",
                "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;WEBP Files (*.webp);;All Files (*)",
            )
            if not chosen:
                return None
            target = Path(chosen)

        comp = np.ascontiguousarray(doc.composite(), dtype=np.uint8)
        pil_img = Image.fromarray(comp, mode="RGBA")
        ext = target.suffix.lower()
        if ext in (".jpg", ".jpeg"):
            rgb = Image.new("RGB", pil_img.size, (255, 255, 255))
            rgb.paste(pil_img, mask=pil_img.split()[3])
            rgb.save(target, format="JPEG", quality=95)
        elif ext == ".webp":
            pil_img.save(target, format="WEBP", quality=95)
        else:
            pil_img.save(target, format="PNG")

        doc.file_path = target
        doc.title = target.name
        doc.modified = False
        self._sync_tab_titles()
        return target

    def close_tab(self, index: int) -> None:
        if not (0 <= index < len(self.tab_manager.sessions)):
            return
        self.tab_manager.close_tab(index)
        self._sync_tabs()

    def close_current_tab(self) -> None:
        if self.tab_manager.active_index >= 0:
            self.close_tab(self.tab_manager.active_index)

    # --- Edit / Clipboard / Undo ---

    def undo(self) -> None:
        doc = self.active_document
        if doc is None:
            return
        if doc.undo() is not None:
            self._refresh_canvas_and_tabs()
            self.layers_panel.refresh()
            self._update_status_bar()

    def redo(self) -> None:
        doc = self.active_document
        if doc is None:
            return
        if doc.redo() is not None:
            self._refresh_canvas_and_tabs()
            self.layers_panel.refresh()
            self._update_status_bar()

    def clear_selection(self) -> None:
        doc = self.active_document
        if doc is None:
            return
        doc.clear_selection()
        self.canvas_view._invalidate_selection_cache()
        self.canvas_view._sync_march_timer()
        self.canvas_view.update()

    def copy_selection(self) -> bool:
        doc = self.active_document
        if doc is None:
            return False
        layer = doc.active_layer
        source = layer.pixels if layer is not None else doc.composite()
        if doc.selection_mask is not None:
            ys, xs = np.nonzero(doc.selection_mask)
            if ys.size == 0:
                return False
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            comp = doc.composite()
            patch = np.ascontiguousarray(comp[y0:y1, x0:x1].copy(), dtype=np.uint8)
            sel_sub = doc.selection_mask[y0:y1, x0:x1] == 0
            patch[sel_sub] = 0
        else:
            patch = np.ascontiguousarray(source.copy(), dtype=np.uint8)

        self._internal_clipboard_rgba = patch
        h, w = int(patch.shape[0]), int(patch.shape[1])
        qimg = QImage(patch.data, w, h, w * 4, QImage.Format_RGBA8888).copy()
        QGuiApplication.clipboard().setImage(qimg)
        return True

    def paste_from_clipboard(self) -> Optional[DocumentSession]:
        """Paste internal selection if present, else quick-create a new tab from OS clipboard image."""
        doc = self.active_document
        if (
            doc is not None
            and doc.selection_mask is not None
            and self._internal_clipboard_rgba is not None
        ):
            new_layer = Layer.from_rgba(self._internal_clipboard_rgba, name="Pasted Layer", copy=True)
            doc.add_layer(layer=new_layer)
            self.layers_panel.refresh()
            self._refresh_canvas_and_tabs()
            return doc

        qimg = QGuiApplication.clipboard().image()
        if qimg.isNull():
            return None

        rgba = qimage_to_rgba_numpy(qimg)
        new_doc = self.tab_manager.new_document_from_rgba(
            rgba,
            title=f"Clipboard-{len(self.tab_manager.sessions) + 1}",
            activate=True,
        )
        self._sync_tabs()
        return new_doc

    # --- Zen Mode ---

    def toggle_zen_mode(self) -> bool:
        """Hide/restore docks, toolbars, menu/options bars, and status bar for full-screen editing."""
        targets: dict[str, QWidget] = {
            "menu_bar": self.menuBar(),
            "tool_options_bar": self.tool_options_bar,
            "tools_toolbar": self.tools_toolbar,
            "layers_dock": self.layers_dock,
            "color_dock": self.color_dock,
            "status_bar": self.statusBar(),
        }

        if not self.is_zen_mode:
            self._pre_zen_visibility = {
                name: (not widget.isHidden()) for name, widget in targets.items()
            }
            for widget in targets.values():
                widget.setVisible(False)
            self.is_zen_mode = True
        else:
            for name, widget in targets.items():
                restore = self._pre_zen_visibility.get(name, True)
                widget.setVisible(restore)
            self.is_zen_mode = False

        if "zen_mode" in self.editor_actions:
            self.editor_actions["zen_mode"].blockSignals(True)
            self.editor_actions["zen_mode"].setChecked(self.is_zen_mode)
            self.editor_actions["zen_mode"].blockSignals(False)

        self.canvas_view.setFocus()
        return self.is_zen_mode

    def _is_focus_in_text_input(self) -> bool:
        fw = QGuiApplication.focusObject()
        return isinstance(fw, (QLineEdit, QAbstractSpinBox, QComboBox, QKeySequenceEdit))

    def focusNextPrevChild(self, next_child: bool) -> bool:
        if self._is_focus_in_text_input():
            return super().focusNextPrevChild(next_child)
        zen_seq = self.keybindings.get("zen_mode", "Tab").strip().lower()
        if zen_seq == "tab":
            return False
        return super().focusNextPrevChild(next_child)

    def eventFilter(self, obj, event: QEvent) -> bool:
        if obj is self.canvas_view and event.type() == QEvent.KeyPress:
            if getattr(event, "key", lambda: None)() == Qt.Key_Tab and not self._is_focus_in_text_input():
                zen_seq = self.keybindings.get("zen_mode", "Tab").strip().lower()
                if zen_seq == "tab":
                    self.toggle_zen_mode()
                    event.accept()
                    return True
        return super().eventFilter(obj, event)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Tab and not self._is_focus_in_text_input():
            zen_seq = self.keybindings.get("zen_mode", "Tab").strip().lower()
            if zen_seq == "tab":
                self.toggle_zen_mode()
                event.accept()
                return
        super().keyPressEvent(event)

    # --- Layer & Image Actions ---

    def center_active_layer(self, horizontal: bool = True, vertical: bool = True) -> None:
        doc = self.active_document
        if doc is None or doc.active_layer is None or doc.active_layer.locked:
            return
        doc.center_active_layer(horizontal=horizontal, vertical=vertical, use_content_bounds=True)
        self._refresh_canvas_and_tabs()

    def trim_transparent(self) -> bool:
        doc = self.active_document
        if doc is None:
            return False
        changed = doc.trim_transparent()
        if changed:
            self._on_document_resized()
        return changed

    def run_ai_background_removal(self) -> None:
        doc = self.active_document
        if doc is None:
            return
        try:
            apply_ai_background_removal(doc)
        except Exception as exc:
            QMessageBox.warning(self, "AI Background Removal", str(exc))
            return
        self.layers_panel.refresh()
        self._refresh_canvas_and_tabs()

    def open_quick_export(self) -> None:
        doc = self.active_document
        if doc is None:
            return
        dlg = QuickExportDialog(doc, parent=self)
        dlg.exec()

    def open_chroma_key(self) -> None:
        doc = self.active_document
        if doc is None:
            return
        dlg = ChromaKeyDialog(doc, parent=self, on_preview=self._refresh_canvas_and_tabs)
        dlg.exec()
        self._refresh_canvas_and_tabs()

    def open_grid_splitter(self) -> None:
        doc = self.active_document
        if doc is None:
            return
        dlg = GridSplitterDialog(doc, self.tab_manager, parent=self)
        dlg.tabs_opened.connect(lambda _docs: self._sync_tabs())
        dlg.exec()
        self._sync_tabs()

    def open_smart_alpha_splitter(self) -> None:
        doc = self.active_document
        if doc is None:
            return
        dlg = SmartAlphaSplitterDialog(
            doc,
            self.tab_manager,
            on_preview_boxes=self.canvas_view.set_sprite_preview_boxes,
            parent=self,
        )
        dlg.tabs_opened.connect(lambda _docs: self._sync_tabs())
        dlg.exec()
        self.canvas_view.set_sprite_preview_boxes([])
        self._sync_tabs()

    def open_batch_processor(self) -> None:
        dlg = BatchProcessorDialog(parent=self)
        dlg.exec()

    def open_hotkey_editor(self) -> None:
        dlg = HotkeyEditorDialog(parent=self)
        dlg.keybindings_saved.connect(self.apply_keybindings)
        dlg.exec()

    # --- Internal Sync Callbacks ---

    def _zoom_by(self, factor: float) -> None:
        doc = self.active_document
        if doc is None:
            return
        cx = self.canvas_view.width() * 0.5
        cy = self.canvas_view.height() * 0.5
        new_zoom = doc.viewport.zoom_at_cursor(cx, cy, factor)
        self._on_zoom_changed(new_zoom)
        self.canvas_view.update()

    def _on_color_picked(self, _color: tuple[int, int, int, int]) -> None:
        self.color_panel.sync_from_context()

    def _refresh_canvas_and_tabs(self) -> None:
        self.canvas_view._invalidate_selection_cache()
        self.canvas_view._sync_march_timer()
        self.canvas_view.update()
        self._sync_tab_titles()
        self._update_status_bar()

    def _on_document_resized(self) -> None:
        self.layers_panel.refresh()
        self._refresh_canvas_and_tabs()

    def _on_canvas_document_modified(self) -> None:
        self._sync_tab_titles()
        self._update_status_bar()

    def _on_layers_changed(self) -> None:
        self._refresh_canvas_and_tabs()

    def _on_active_layer_changed(self, _idx: int) -> None:
        self.canvas_view.update()

    def _on_tool_options_changed(self) -> None:
        self.canvas_view.update()

    def _on_tool_action_requested(self, action: str) -> None:
        doc = self.active_document
        tool = self.canvas_view.active_tool
        if doc is None or tool is None:
            return
        key = Qt.Key_Return if action == "apply" else Qt.Key_Escape
        if tool.key_press(doc, key, 0):
            self.layers_panel.refresh()
            self._refresh_canvas_and_tabs()

    def _on_cursor_moved(self, cx: float, cy: float) -> None:
        self.status_cursor_label.setText(f"X: {int(round(cx))}, Y: {int(round(cy))}")

    def _on_zoom_changed(self, zoom: float) -> None:
        self.status_zoom_label.setText(f"Zoom: {int(round(zoom * 100.0))}%")

    def _update_status_bar(self) -> None:
        doc = self.active_document
        if doc is None:
            self.status_zoom_label.setText("Zoom: --")
            self.status_doc_label.setText("No document")
        else:
            self.status_zoom_label.setText(f"Zoom: {int(round(doc.viewport.zoom * 100.0))}%")
            self.status_doc_label.setText(f"{doc.width} x {doc.height} px")
