"""Main tabbed workspace window integrating CanvasView, panels, tools, menus, dialogs, and Zen mode."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import (
    QAction,
    QActionGroup,
    QDragEnterEvent,
    QDragMoveEvent,
    QDropEvent,
    QGuiApplication,
    QImage,
    QKeySequence,
)
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDockWidget,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QKeySequenceEdit,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QSpinBox,
    QTabBar,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app.config.i18n import get_language, set_language as set_i18n_language, tr
from app.core.ai_bg_remove import apply_ai_background_removal
from app.core.document import CallableCommand, DocumentSession, DocumentTabManager
from app.core.layer import Layer
from app.tools import BaseTool, CropTool, ToolContext, ToolType, create_default_tools
from app.tools.base_tool import align_selection_mask_to_layer
from app.ui.canvas_view import CanvasView, extract_image_paths_from_mime
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


_TOOL_ORDER: tuple[str, ...] = (
    ToolType.MOVE.value,
    ToolType.HAND.value,
    ToolType.MARQUEE.value,
    ToolType.LASSO.value,
    ToolType.MAGIC_WAND.value,
    ToolType.CROP.value,
    ToolType.TRANSFORM.value,
    ToolType.BRUSH.value,
    ToolType.ERASER.value,
    ToolType.SMUDGE.value,
    ToolType.BLUR_SHARPEN.value,
    ToolType.BUCKET_FILL.value,
    ToolType.GRADIENT.value,
    ToolType.EYEDROPPER.value,
    ToolType.TEXT.value,
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


def load_image_file_rgba(path: Path) -> np.ndarray:
    """Load an image file from disk as a contiguous (H, W, 4) uint8 RGBA array."""
    with Image.open(path) as im:
        return np.ascontiguousarray(np.asarray(im.convert("RGBA"), dtype=np.uint8))


class _NewCanvasDialog(QDialog):
    """Lightweight dialog for custom document dimensions and background fill."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("new_canvas.title"))
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.width_spin = QSpinBox()
        self.width_spin.setRange(1, 16384)
        self.width_spin.setValue(1024)
        self.height_spin = QSpinBox()
        self.height_spin.setRange(1, 16384)
        self.height_spin.setValue(1024)
        self.bg_combo = QComboBox()
        self.bg_combo.addItem(tr("new_canvas.bg_white"), "white")
        self.bg_combo.addItem(tr("new_canvas.bg_transparent"), "transparent")
        self.bg_combo.addItem(tr("new_canvas.bg_black"), "black")
        form.addRow(tr("new_canvas.width"), self.width_spin)
        form.addRow(tr("new_canvas.height"), self.height_spin)
        form.addRow(tr("new_canvas.background"), self.bg_combo)
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
        self.setWindowTitle(tr("app.title"))
        self.resize(1280, 820)
        self.setAcceptDrops(True)

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
        self._misc_actions: dict[str, QAction] = {}
        self._menus: dict[str, object] = {}
        self.keybindings: dict[str, str] = {}

        self._build_central_area()
        self._build_docks()
        self._build_tools_toolbar()
        self._build_menus()
        self._build_status_bar()
        self._connect_signals()

        self.retranslate_ui()
        self.keybindings = load_keybindings()
        self.apply_keybindings(self.keybindings)

        if create_initial_doc and not self.tab_manager.sessions:
            self.new_document(800, 600, initial_fill=(255, 255, 255, 255))
        else:
            self._sync_tabs()

        self.select_tool(ToolType.BRUSH.value)
        self.canvas_view.installEventFilter(self)

    def _build_central_area(self) -> None:
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        tab_strip = QWidget()
        tab_strip.setObjectName("TabStrip")
        tab_strip_layout = QHBoxLayout(tab_strip)
        tab_strip_layout.setContentsMargins(0, 0, 0, 0)
        tab_strip_layout.setSpacing(0)

        self.tab_bar = QTabBar()
        self.tab_bar.setTabsClosable(True)
        self.tab_bar.setMovable(False)
        self.tab_bar.setExpanding(False)
        self.tab_bar.setDocumentMode(True)
        tab_strip_layout.addWidget(self.tab_bar)

        self.new_tab_btn = QToolButton()
        self.new_tab_btn.setObjectName("NewTabButton")
        self.new_tab_btn.setText("+")
        self.new_tab_btn.clicked.connect(lambda: self.new_document())
        tab_strip_layout.addWidget(self.new_tab_btn)
        tab_strip_layout.addStretch(1)

        layout.addWidget(tab_strip)

        self.tool_options_bar = ToolOptionsBar(self.tool_context)
        layout.addWidget(self.tool_options_bar)

        self.canvas_view = CanvasView(tool_context=self.tool_context)
        self.canvas_view.set_hand_tool(self.tools.get(ToolType.HAND.value))
        layout.addWidget(self.canvas_view, 1)

        self.setCentralWidget(central)

    def _build_docks(self) -> None:
        self.color_panel = ColorPanel(self.tool_context)
        self.color_dock = QDockWidget(tr("dock.colors"), self)
        self.color_dock.setObjectName("ColorDock")
        self.color_dock.setWidget(self.color_panel)
        self.color_dock.setMinimumWidth(240)
        self.addDockWidget(Qt.RightDockWidgetArea, self.color_dock)

        self.layers_panel = LayersPanel(self.tool_context)
        self.layers_dock = QDockWidget(tr("dock.layers"), self)
        self.layers_dock.setObjectName("LayersDock")
        self.layers_dock.setWidget(self.layers_panel)
        self.layers_dock.setMinimumWidth(240)
        self.addDockWidget(Qt.RightDockWidgetArea, self.layers_dock)

    def _build_tools_toolbar(self) -> None:
        self.tools_toolbar = QToolBar(tr("dock.tools"), self)
        self.tools_toolbar.setObjectName("ToolsToolBar")
        self.tools_toolbar.setMovable(False)
        self.tools_toolbar.setOrientation(Qt.Vertical)
        self.addToolBar(Qt.LeftToolBarArea, self.tools_toolbar)

        self._tool_action_group = QActionGroup(self)
        self._tool_action_group.setExclusive(True)

        for tool_id in _TOOL_ORDER:
            act = QAction("", self)
            act.setCheckable(True)
            act.setData(tool_id)
            act.triggered.connect(lambda _chk=False, tid=tool_id: self.select_tool(tid))
            self._tool_action_group.addAction(act)
            self.tools_toolbar.addAction(act)
            self.tool_actions[tool_id] = act

    def _register_editor_action(
        self,
        key: str,
        text_key: str,
        slot,
        checkable: bool = False,
    ) -> QAction:
        act = QAction(tr(text_key), self)
        act.setCheckable(checkable)
        act.triggered.connect(slot)
        self.addAction(act)
        self.editor_actions[key] = act
        self._misc_actions[text_key] = act
        return act

    def _register_misc_action(self, text_key: str, slot, shortcut: str = "") -> QAction:
        act = QAction(tr(text_key), self)
        if shortcut:
            act.setShortcut(QKeySequence(shortcut))
        act.triggered.connect(slot)
        self._misc_actions[text_key] = act
        return act

    def _build_menus(self) -> None:
        mb = self.menuBar()

        # File Menu
        file_menu = mb.addMenu(tr("menu.file"))
        self._menus["menu.file"] = file_menu
        file_menu.addAction(self._register_editor_action("new_tab", "action.new_tab", self.new_document))
        file_menu.addAction(self._register_misc_action("action.new_custom", self._open_new_custom_dialog))
        file_menu.addAction(self._register_editor_action("open_file", "action.open_file", self.open_file))
        file_menu.addAction(self._register_editor_action("save_file", "action.save_file", self.save_file))
        file_menu.addSeparator()
        file_menu.addAction(
            self._register_editor_action("quick_export", "action.quick_export", self.open_quick_export)
        )
        file_menu.addAction(self._register_misc_action("action.batch", self.open_batch_processor))
        file_menu.addSeparator()
        file_menu.addAction(self._register_misc_action("action.close_tab", self.close_current_tab, "Ctrl+W"))
        file_menu.addAction(self._register_misc_action("action.exit", self.close))

        # Edit Menu
        edit_menu = mb.addMenu(tr("menu.edit"))
        self._menus["menu.edit"] = edit_menu
        edit_menu.addAction(self._register_editor_action("undo", "action.undo", self.undo))
        edit_menu.addAction(self._register_editor_action("redo", "action.redo", self.redo))
        edit_menu.addSeparator()
        edit_menu.addAction(self._register_misc_action("action.copy", self.copy_selection, "Ctrl+C"))
        edit_menu.addAction(
            self._register_editor_action("paste_clipboard", "action.paste", self.paste_from_clipboard)
        )
        edit_menu.addAction(self._register_misc_action("action.deselect", self.clear_selection, "Ctrl+D"))
        edit_menu.addAction(self._register_misc_action("ctx.invert_sel", self.invert_selection, "Ctrl+Shift+I"))
        edit_menu.addAction(self._register_misc_action("ctx.clear_sel_contents", self.delete_selection_contents, "Delete"))
        edit_menu.addSeparator()
        edit_menu.addAction(self._register_misc_action("action.hotkeys", self.open_hotkey_editor))

        # Image Menu
        image_menu = mb.addMenu(tr("menu.image"))
        self._menus["menu.image"] = image_menu
        image_menu.addAction(self._register_misc_action("action.trim", self.trim_transparent))
        image_menu.addSeparator()
        image_menu.addAction(self._register_misc_action("action.ai_bg", self.run_ai_background_removal))
        image_menu.addAction(self._register_misc_action("action.chroma_key", self.open_chroma_key))
        image_menu.addSeparator()
        image_menu.addAction(self._register_misc_action("action.grid_split", self.open_grid_splitter))
        image_menu.addAction(self._register_misc_action("action.alpha_split", self.open_smart_alpha_splitter))

        # Layer Menu
        layer_menu = mb.addMenu(tr("menu.layer"))
        self._menus["menu.layer"] = layer_menu
        layer_menu.addAction(self._register_misc_action("action.add_layer", self.layers_panel.add_layer))
        layer_menu.addAction(self._register_misc_action("action.dup_layer", self.layers_panel.duplicate_layer))
        layer_menu.addAction(self._register_misc_action("ctx.layer_via_copy", self.layer_via_copy, "Ctrl+J"))
        layer_menu.addAction(self._register_misc_action("ctx.layer_via_cut", self.layer_via_cut, "Ctrl+Shift+J"))
        layer_menu.addAction(self._register_misc_action("action.del_layer", self.layers_panel.delete_layer))
        layer_menu.addAction(self._register_misc_action("action.merge_down", self.layers_panel.merge_down))
        layer_menu.addSeparator()
        layer_menu.addAction(
            self._register_misc_action("action.toggle_mask_add_del", self.layers_panel.add_or_delete_mask)
        )
        layer_menu.addAction(
            self._register_misc_action("action.toggle_clip", self.layers_panel.toggle_clipping_mask)
        )
        layer_menu.addSeparator()
        layer_menu.addAction(
            self._register_misc_action("action.center_h", lambda: self.center_active_layer(True, False))
        )
        layer_menu.addAction(
            self._register_misc_action("action.center_v", lambda: self.center_active_layer(False, True))
        )
        layer_menu.addAction(
            self._register_editor_action(
                "center_both", "action.center_both", lambda: self.center_active_layer(True, True)
            )
        )

        # Tools Menu
        tools_menu = mb.addMenu(tr("menu.tools"))
        self._menus["menu.tools"] = tools_menu
        for tool_id in _TOOL_ORDER:
            if tool_id in self.tool_actions:
                tools_menu.addAction(self.tool_actions[tool_id])
        tools_menu.addSeparator()
        tools_menu.addAction(
            self._register_misc_action("action.swap_colors", self.color_panel.swap_colors, "X")
        )
        tools_menu.addAction(
            self._register_misc_action("action.reset_colors", self.color_panel.reset_colors, "D")
        )

        # View Menu
        view_menu = mb.addMenu(tr("menu.view"))
        self._menus["menu.view"] = view_menu
        view_menu.addAction(
            self._register_misc_action("action.zoom_in", lambda: self._zoom_by(1.25), "Ctrl+=")
        )
        view_menu.addAction(
            self._register_misc_action("action.zoom_out", lambda: self._zoom_by(0.8), "Ctrl+-")
        )
        view_menu.addAction(
            self._register_misc_action("action.fit_screen", self.canvas_view.fit_to_view, "Ctrl+0")
        )
        view_menu.addAction(
            self._register_misc_action("action.actual_size", self.canvas_view.set_zoom_100, "Ctrl+1")
        )
        view_menu.addSeparator()
        view_menu.addAction(
            self._register_editor_action("zen_mode", "action.zen_mode", self.toggle_zen_mode, checkable=True)
        )
        view_menu.addSeparator()
        view_menu.addAction(self.layers_dock.toggleViewAction())
        view_menu.addAction(self.color_dock.toggleViewAction())

        # Language submenu
        lang_menu = view_menu.addMenu(tr("menu.language"))
        self._menus["menu.language"] = lang_menu
        self._lang_group = QActionGroup(self)
        self._lang_group.setExclusive(True)
        self.act_lang_ru = QAction(tr("lang.ru"), self)
        self.act_lang_ru.setCheckable(True)
        self.act_lang_ru.setChecked(get_language() == "ru")
        self.act_lang_ru.triggered.connect(lambda: self.set_language("ru"))
        self.act_lang_en = QAction(tr("lang.en"), self)
        self.act_lang_en.setCheckable(True)
        self.act_lang_en.setChecked(get_language() == "en")
        self.act_lang_en.triggered.connect(lambda: self.set_language("en"))
        self._lang_group.addAction(self.act_lang_ru)
        self._lang_group.addAction(self.act_lang_en)
        lang_menu.addAction(self.act_lang_ru)
        lang_menu.addAction(self.act_lang_en)

    def _build_status_bar(self) -> None:
        sb = self.statusBar()
        self.status_zoom_label = QLabel()
        self.status_zoom_label.setObjectName("StatusZoomLabel")
        self.status_cursor_label = QLabel(tr("status.cursor", x=0, y=0))
        self.status_doc_label = QLabel()

        sb.addWidget(self.status_zoom_label)
        sb.addWidget(self.status_cursor_label)
        sb.addPermanentWidget(self.status_doc_label)

    def _connect_signals(self) -> None:
        self.tab_bar.currentChanged.connect(self._on_tab_changed)
        self.tab_bar.tabCloseRequested.connect(self.close_tab)

        self.canvas_view.cursorMoved.connect(self._on_cursor_moved)
        self.canvas_view.zoomChanged.connect(self._on_zoom_changed)
        self.canvas_view.documentModified.connect(self._on_canvas_document_modified)
        self.canvas_view.filesDropped.connect(self.insert_image_files_as_layers)
        self.canvas_view.contextMenuRequested.connect(self.show_canvas_context_menu)

        self.layers_panel.layersChanged.connect(self._on_layers_changed)
        self.layers_panel.activeLayerChanged.connect(self._on_active_layer_changed)
        self.layers_panel.filesDropped.connect(self.insert_image_files_as_layers)

        self.tool_options_bar.optionsChanged.connect(self._on_tool_options_changed)
        self.tool_options_bar.cropOptionsChanged.connect(self._on_crop_options_changed)
        self.tool_options_bar.centerRequested.connect(self.center_active_layer)
        self.tool_options_bar.clearSelectionRequested.connect(self.clear_selection)
        self.tool_options_bar.fitViewRequested.connect(self.canvas_view.fit_to_view)
        self.tool_options_bar.zoom100Requested.connect(self.canvas_view.set_zoom_100)
        self.tool_options_bar.toolActionRequested.connect(self._on_tool_action_requested)

        self.color_panel.fgColorChanged.connect(lambda _c: self.canvas_view.update())
        self.color_panel.bgColorChanged.connect(lambda _c: self.canvas_view.update())

    def set_language(self, lang: str, persist: bool = True) -> None:
        set_i18n_language(lang, persist=persist)
        self.act_lang_ru.setChecked(get_language() == "ru")
        self.act_lang_en.setChecked(get_language() == "en")
        self.retranslate_ui()

    def retranslate_ui(self) -> None:
        self.setWindowTitle(tr("app.title"))
        self.color_dock.setWindowTitle(tr("dock.colors"))
        self.layers_dock.setWindowTitle(tr("dock.layers"))
        self.tools_toolbar.setWindowTitle(tr("dock.tools"))
        self.new_tab_btn.setToolTip(tr("tab.new_btn_tip"))

        for key, menu in self._menus.items():
            menu.setTitle(tr(key))  # type: ignore[attr-defined]

        for text_key, act in self._misc_actions.items():
            act.setText(tr(text_key))

        for tool_id, act in self.tool_actions.items():
            act.setText(tr(f"tool.short.{tool_id}"))
            full_name = tr(f"tool.name.{tool_id}")
            seq = self.keybindings.get(tool_id, "")
            act.setToolTip(f"{full_name} ({seq})" if seq else full_name)

        self.tool_options_bar.retranslate_ui()
        self.tool_options_bar.set_active_tool(self.active_tool_id)
        self.color_panel.retranslate_ui()
        self.layers_panel.retranslate_ui()
        self._update_status_bar()
        self.canvas_view.update()

    def apply_keybindings(self, bindings: dict[str, str]) -> None:
        self.keybindings = dict(bindings)
        for tool_id, act in self.tool_actions.items():
            seq = bindings.get(tool_id, "")
            act.setShortcut(QKeySequence.fromString(seq) if seq else QKeySequence())
            full_name = tr(f"tool.name.{tool_id}")
            act.setToolTip(f"{full_name} ({seq})" if seq else full_name)
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
        self.tool_options_bar.set_active_tool(tool_id, display_name=tr(f"tool.name.{tool_id}"))
        self.layers_panel.refresh()

    # --- Drag & Drop onto MainWindow (Tab bar / empty shell -> new tabs) ---

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if extract_image_paths_from_mime(event.mimeData()):
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:
        if extract_image_paths_from_mime(event.mimeData()):
            event.acceptProposedAction()
            return
        super().dragMoveEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:
        paths = extract_image_paths_from_mime(event.mimeData())
        if paths:
            for p in paths:
                self.open_file(p)
            event.acceptProposedAction()
            return
        super().dropEvent(event)

    def insert_image_files_as_layers(self, paths: list[Path]) -> list[Layer]:
        """Insert dropped/pasted local image files as centered layers in the active document."""
        doc = self.active_document
        if doc is None:
            for p in paths:
                self.open_file(p)
            return []

        inserted: list[Layer] = []
        for path in paths:
            try:
                rgba = load_image_file_rgba(Path(path))
            except Exception:
                continue
            new_layer = Layer.from_rgba(rgba, name=Path(path).stem, copy=False)
            new_layer.center_on_canvas(doc.width, doc.height, use_content_bounds=False)
            doc.add_layer(layer=new_layer, index=doc.active_layer_index + 1)
            inserted.append(new_layer)

        if inserted:
            self.layers_panel.refresh()
            self._refresh_canvas_and_tabs()
        return inserted

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
        initial_fill: tuple[int, int, int, int] = (255, 255, 255, 255),
    ) -> DocumentSession:
        if isinstance(width, bool):
            width = 800
        doc_title = title or f"{tr('doc.untitled')}-{len(self.tab_manager.sessions) + 1}"
        doc = self.tab_manager.new_document(
            width=int(width),
            height=int(height),
            title=doc_title,
            initial_fill=initial_fill,
            activate=True,
        )
        if doc.layers:
            doc.layers[0].name = tr("doc.background_layer")
        doc.modified = False
        self._sync_tabs()
        if self.canvas_view.isVisible() and self.canvas_view.width() > 120 and self.canvas_view.height() > 120:
            self.canvas_view.fit_to_view()
        return doc

    def _open_new_custom_dialog(self) -> None:
        dlg = _NewCanvasDialog(self)
        if dlg.exec() == QDialog.Accepted:
            fill_map = {
                "transparent": (0, 0, 0, 0),
                "white": (255, 255, 255, 255),
                "black": (0, 0, 0, 255),
            }
            bg_key = str(dlg.bg_combo.currentData() or "white")
            fill = fill_map.get(bg_key, (255, 255, 255, 255))
            self.new_document(dlg.width_spin.value(), dlg.height_spin.value(), initial_fill=fill)

    def open_file(self, file_path: Optional[Path] = None) -> Optional[DocumentSession]:
        if file_path is None or isinstance(file_path, bool):
            chosen, _ = QFileDialog.getOpenFileName(
                self,
                tr("dlg.open_title"),
                "",
                "Images (*.png *.jpg *.jpeg *.webp *.bmp);;All Files (*)",
            )
            if not chosen:
                return None
            target = Path(chosen)
        else:
            target = Path(file_path)

        rgba = load_image_file_rgba(target)
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
                tr("dlg.save_title"),
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

    def invert_selection(self) -> None:
        doc = self.active_document
        if doc is None:
            return
        if doc.selection_mask is None:
            inverted = np.full((doc.height, doc.width), 255, dtype=np.uint8)
        else:
            inverted = np.ascontiguousarray(255 - doc.selection_mask, dtype=np.uint8)
        doc.set_selection_mask(inverted if np.any(inverted) else None)
        self._refresh_canvas_and_tabs()

    def _extract_selected_region_for_layer(
        self,
        doc: DocumentSession,
        layer: Layer,
    ) -> Optional[tuple[int, int, int, int, np.ndarray, np.ndarray]]:
        """Return (ix0, iy0, ix1, iy1, sel_sub_bool, patch_rgba) intersecting layer and selection."""
        if doc.selection_mask is None:
            return None
        ys, xs = np.nonzero(doc.selection_mask)
        if ys.size == 0:
            return None
        sx0, sx1 = int(xs.min()), int(xs.max()) + 1
        sy0, sy1 = int(ys.min()), int(ys.max()) + 1

        ix0 = max(sx0, layer.offset_x)
        iy0 = max(sy0, layer.offset_y)
        ix1 = min(sx1, layer.offset_x + layer.width)
        iy1 = min(sy1, layer.offset_y + layer.height)
        if ix1 <= ix0 or iy1 <= iy0:
            return None

        sel_sub = doc.selection_mask[iy0:iy1, ix0:ix1] > 0
        if not np.any(sel_sub):
            return None

        lx0 = ix0 - layer.offset_x
        ly0 = iy0 - layer.offset_y
        lx1 = ix1 - layer.offset_x
        ly1 = iy1 - layer.offset_y
        patch = np.ascontiguousarray(layer.pixels[ly0:ly1, lx0:lx1].copy(), dtype=np.uint8)
        patch[~sel_sub] = 0
        return (ix0, iy0, ix1, iy1, sel_sub, patch)

    def layer_via_copy(self) -> Optional[Layer]:
        doc = self.active_document
        if doc is None or doc.active_layer is None:
            return None
        layer = doc.active_layer
        if doc.selection_mask is None:
            copied = doc.duplicate_layer(doc.active_layer_index)
            self.layers_panel.refresh()
            self._refresh_canvas_and_tabs()
            return copied

        extracted = self._extract_selected_region_for_layer(doc, layer)
        if extracted is None:
            return None
        ix0, iy0, _ix1, _iy1, _sel_sub, patch = extracted
        new_layer = Layer.from_rgba(
            patch,
            name=f"{layer.name} (Copy)",
            copy=False,
            offset_x=ix0,
            offset_y=iy0,
        )
        doc.add_layer(layer=new_layer, index=doc.active_layer_index + 1)
        doc.clear_selection()
        self.layers_panel.refresh()
        self._refresh_canvas_and_tabs()
        return new_layer

    def layer_via_cut(self) -> Optional[Layer]:
        doc = self.active_document
        if doc is None or doc.active_layer is None or doc.active_layer.locked:
            return None
        layer = doc.active_layer
        if doc.selection_mask is None:
            return self.layer_via_copy()

        extracted = self._extract_selected_region_for_layer(doc, layer)
        if extracted is None:
            return None
        ix0, iy0, ix1, iy1, sel_sub, patch = extracted
        lx0 = ix0 - layer.offset_x
        ly0 = iy0 - layer.offset_y
        lw = ix1 - ix0
        lh = iy1 - iy0

        before_patch = layer.get_patch((lx0, ly0, lw, lh))
        cleared_patch = before_patch.copy()
        cleared_patch[sel_sub] = 0

        insert_idx = doc.active_layer_index + 1
        prev_active = doc.active_layer_index
        prev_sel = doc.selection_mask
        new_layer = Layer.from_rgba(
            patch,
            name=f"{layer.name} (Cut)",
            copy=False,
            offset_x=ix0,
            offset_y=iy0,
        )

        def do(d: DocumentSession) -> None:
            layer.set_patch((lx0, ly0, lw, lh), cleared_patch)
            d.layers.insert(insert_idx, new_layer)
            d.active_layer_index = insert_idx
            d.selection_mask = None

        def undo(d: DocumentSession) -> None:
            d.layers.pop(insert_idx)
            layer.set_patch((lx0, ly0, lw, lh), before_patch)
            d.active_layer_index = prev_active
            d.selection_mask = prev_sel

        do(doc)
        doc.undo_stack.push(
            CallableCommand(
                undo_fn=undo,
                redo_fn=do,
                description="Layer via Cut",
                estimated_bytes=int(before_patch.nbytes + patch.nbytes),
            )
        )
        doc.modified = True
        doc.invalidate_composite()
        self.layers_panel.refresh()
        self._refresh_canvas_and_tabs()
        return new_layer

    def fill_selection_with_fg(self) -> bool:
        doc = self.active_document
        if doc is None or doc.active_layer is None or doc.active_layer.locked:
            return False
        layer = doc.active_layer
        r, g, b, a = self.tool_context.fg_color
        fill_rgba = np.array([r, g, b, a], dtype=np.uint8)

        if doc.selection_mask is None:
            with doc.edit_layer_context(layer=layer, description="Fill Layer"):
                layer.pixels[:, :] = fill_rgba
        else:
            aligned = align_selection_mask_to_layer(doc, layer)
            if aligned is None or not np.any(aligned > 0):
                return False
            with doc.edit_layer_context(layer=layer, description="Fill Selection"):
                layer.pixels[aligned > 0] = fill_rgba

        self.layers_panel.refresh()
        self._refresh_canvas_and_tabs()
        return True

    def delete_selection_contents(self) -> bool:
        doc = self.active_document
        if doc is None or doc.active_layer is None or doc.active_layer.locked:
            return False
        layer = doc.active_layer
        if doc.selection_mask is None:
            with doc.edit_layer_context(layer=layer, description="Clear Layer"):
                layer.pixels[:, :] = 0
        else:
            aligned = align_selection_mask_to_layer(doc, layer)
            if aligned is None or not np.any(aligned > 0):
                return False
            with doc.edit_layer_context(layer=layer, description="Clear Selection"):
                layer.pixels[aligned > 0] = 0

        self.layers_panel.refresh()
        self._refresh_canvas_and_tabs()
        return True

    def crop_to_selection(self) -> bool:
        doc = self.active_document
        if doc is None or doc.selection_mask is None:
            return False
        ys, xs = np.nonzero(doc.selection_mask)
        if ys.size == 0:
            return False
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        cw, ch = x1 - x0, y1 - y0
        if cw <= 0 or ch <= 0:
            return False

        old_w, old_h = doc.width, doc.height
        layer_offsets = [(l, l.offset_x, l.offset_y) for l in doc.layers]
        old_sel = doc.selection_mask

        def do(d: DocumentSession) -> None:
            d.width = cw
            d.height = ch
            for l, ox, oy in layer_offsets:
                l.offset_x = ox - x0
                l.offset_y = oy - y0
            d.selection_mask = None
            d._composited_cache = None
            d._clip_alpha_scratch = None

        def undo(d: DocumentSession) -> None:
            d.width = old_w
            d.height = old_h
            for l, ox, oy in layer_offsets:
                l.offset_x = ox
                l.offset_y = oy
            d.selection_mask = old_sel
            d._composited_cache = None
            d._clip_alpha_scratch = None

        do(doc)
        doc.undo_stack.push(
            CallableCommand(undo_fn=undo, redo_fn=do, description=f"Crop to Selection ({cw}x{ch})")
        )
        doc.modified = True
        doc.invalidate_composite()
        self._on_document_resized()
        return True

    def build_canvas_context_menu(self) -> QMenu:
        """Build a Photoshop-style context menu tailored to active selection state."""
        menu = QMenu(self)
        doc = self.active_document
        if doc is None:
            menu.addAction(tr("ctx.paste"), self.paste_from_clipboard)
            return menu

        if doc.selection_mask is not None:
            menu.addAction(tr("ctx.deselect"), self.clear_selection)
            menu.addAction(tr("ctx.invert_sel"), self.invert_selection)
            menu.addSeparator()
            menu.addAction(tr("ctx.layer_via_copy"), self.layer_via_copy)
            menu.addAction(tr("ctx.layer_via_cut"), self.layer_via_cut)
            menu.addAction(tr("ctx.free_transform"), lambda: self.select_tool(ToolType.TRANSFORM.value))
            menu.addSeparator()
            menu.addAction(tr("ctx.fill_fg"), self.fill_selection_with_fg)
            menu.addAction(tr("ctx.clear_sel_contents"), self.delete_selection_contents)
            menu.addAction(tr("ctx.crop_to_sel"), self.crop_to_selection)
        else:
            menu.addAction(tr("ctx.free_transform"), lambda: self.select_tool(ToolType.TRANSFORM.value))
            menu.addAction(tr("ctx.dup_layer"), self.layers_panel.duplicate_layer)
            menu.addAction(tr("ctx.center_canvas"), lambda: self.center_active_layer(True, True))
            menu.addAction(tr("ctx.paste"), self.paste_from_clipboard)
        return menu

    def show_canvas_context_menu(self, global_pos) -> None:
        menu = self.build_canvas_context_menu()
        menu.exec(global_pos)

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
        """Paste internal selection, Explorer image URLs, or OS clipboard screenshot."""
        doc = self.active_document
        if (
            doc is not None
            and doc.selection_mask is not None
            and self._internal_clipboard_rgba is not None
        ):
            new_layer = Layer.from_rgba(
                self._internal_clipboard_rgba, name=tr("doc.layer_pasted"), copy=True
            )
            doc.add_layer(layer=new_layer)
            self.layers_panel.refresh()
            self._refresh_canvas_and_tabs()
            return doc

        clipboard = QGuiApplication.clipboard()
        mime = clipboard.mimeData()
        url_paths = extract_image_paths_from_mime(mime)
        if url_paths:
            if doc is not None:
                self.insert_image_files_as_layers(url_paths)
                return doc
            opened_doc: Optional[DocumentSession] = None
            for p in url_paths:
                opened_doc = self.open_file(p)
            return opened_doc

        qimg = clipboard.image()
        if qimg.isNull():
            return None

        rgba = qimage_to_rgba_numpy(qimg)
        new_doc = self.tab_manager.new_document_from_rgba(
            rgba,
            title=f"{tr('doc.clipboard_prefix')}-{len(self.tab_manager.sessions) + 1}",
            activate=True,
        )
        new_doc.modified = False
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
            QMessageBox.warning(self, tr("action.ai_bg").replace("&", ""), str(exc))
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

    def _on_crop_options_changed(self) -> None:
        crop_tool = self.tools.get(ToolType.CROP.value)
        if isinstance(crop_tool, CropTool):
            crop_tool.apply_aspect_ratio_to_current_rect(self.active_document)

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
        self.status_cursor_label.setText(
            tr("status.cursor", x=int(round(cx)), y=int(round(cy)))
        )

    def _on_zoom_changed(self, zoom: float) -> None:
        self.status_zoom_label.setText(
            tr("status.zoom", pct=int(round(zoom * 100.0)))
        )

    def _update_status_bar(self) -> None:
        doc = self.active_document
        if doc is None:
            self.status_zoom_label.setText(tr("status.zoom_empty"))
            self.status_doc_label.setText(tr("status.no_doc"))
        else:
            self.status_zoom_label.setText(
                tr("status.zoom", pct=int(round(doc.viewport.zoom * 100.0)))
            )
            self.status_doc_label.setText(
                tr("status.doc_size", w=doc.width, h=doc.height)
            )
