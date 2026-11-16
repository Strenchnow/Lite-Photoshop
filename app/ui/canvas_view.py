"""High-FPS QWidget viewport for rendering DocumentSession, overlays, zoom-to-cursor, and panning."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import QLineF, QMimeData, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QContextMenuEvent,
    QDragEnterEvent,
    QDragMoveEvent,
    QDropEvent,
    QFont,
    QImage,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PySide6.QtWidgets import QWidget

from app.config.i18n import tr
from app.core.c_bridge import ComponentBox
from app.core.document import DocumentSession
from app.tools.base_tool import BaseTool, ToolContext, extract_modifiers


SUPPORTED_IMAGE_EXTENSIONS: frozenset[str] = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp"})


def extract_image_paths_from_mime(mime_data: Optional[QMimeData]) -> list[Path]:
    """Extract existing local image file paths from QMimeData URLs."""
    if mime_data is None or not mime_data.hasUrls():
        return []
    result: list[Path] = []
    for url in mime_data.urls():
        if not url.isLocalFile():
            continue
        p = Path(url.toLocalFile())
        if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            result.append(p)
    return result


_CURSOR_MAP: dict[str, Qt.CursorShape] = {
    "arrow": Qt.ArrowCursor,
    "crosshair": Qt.CrossCursor,
    "cross": Qt.CrossCursor,
    "hand": Qt.OpenHandCursor,
    "open_hand": Qt.OpenHandCursor,
    "closed_hand": Qt.ClosedHandCursor,
    "move": Qt.SizeAllCursor,
    "size_all": Qt.SizeAllCursor,
    "text": Qt.IBeamCursor,
    "ibeam": Qt.IBeamCursor,
    "eyedropper": Qt.CrossCursor,
    "picker": Qt.CrossCursor,
}


def _qt_flag_int(flag) -> int:
    val = getattr(flag, "value", flag)
    return int(val)


def _create_checkerboard_brush(tile_size: int = 8) -> QBrush:
    size = tile_size * 2
    pm = QPixmap(size, size)
    pm.fill(QColor(42, 42, 46))
    p = QPainter(pm)
    light = QColor(56, 56, 62)
    p.fillRect(0, 0, tile_size, tile_size, light)
    p.fillRect(tile_size, tile_size, tile_size, tile_size, light)
    p.end()
    return QBrush(pm)


class CanvasView(QWidget):
    """Hardware-friendly QPainter viewport with Zoom-to-Cursor, temp panning, guides, and tool overlays."""

    cursorMoved = Signal(float, float)
    zoomChanged = Signal(float)
    documentModified = Signal()
    filesDropped = Signal(list)
    contextMenuRequested = Signal(object)

    def __init__(
        self,
        document: Optional[DocumentSession] = None,
        tool_context: Optional[ToolContext] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self.setAcceptDrops(True)

        self._document: Optional[DocumentSession] = document
        self._tool_context: Optional[ToolContext] = tool_context
        self._active_tool: Optional[BaseTool] = None
        self._hand_tool: Optional[BaseTool] = None
        self._sprite_preview_boxes: list[ComponentBox] = []
        self._checker_brush: Optional[QBrush] = None

        # Temporary Hand panning state (Space+LeftDrag or MiddleDrag from any tool)
        self._space_pressed: bool = False
        self._temp_pan_active: bool = False
        self._temp_pan_button: Optional[Qt.MouseButton] = None
        self._saved_tool_before_pan: Optional[BaseTool] = None
        self._last_pan_pos: Optional[QPointF] = None

        # Selection outline animation & vectorized contour cache
        self._march_offset: int = 0
        self._march_timer = QTimer(self)
        self._march_timer.setInterval(160)
        self._march_timer.timeout.connect(self._advance_march)
        self._sel_cached_ref: Optional[np.ndarray] = None
        self._sel_bbox: Optional[tuple[int, int, int, int]] = None
        self._sel_is_rect: bool = True
        self._sel_lines: list[QLineF] = []

        self._sync_march_timer()

    @property
    def document(self) -> Optional[DocumentSession]:
        return self._document

    def set_document(self, doc: Optional[DocumentSession], reset_view: bool = False) -> None:
        self._end_temp_pan(force_restore=True)
        self._document = doc
        self._sprite_preview_boxes = []
        self._invalidate_selection_cache()
        if doc is not None and reset_view and self.width() > 0 and self.height() > 0:
            doc.viewport.reset_view(self.width(), self.height(), doc.width, doc.height)
            self.zoomChanged.emit(doc.viewport.zoom)
        self._sync_march_timer()
        self.update()

    @property
    def tool_context(self) -> Optional[ToolContext]:
        return self._tool_context

    def set_tool_context(self, context: Optional[ToolContext]) -> None:
        self._tool_context = context
        self.update()

    @property
    def active_tool(self) -> Optional[BaseTool]:
        return self._active_tool

    def set_active_tool(self, tool: Optional[BaseTool]) -> None:
        if self._temp_pan_active or self._space_pressed:
            self._saved_tool_before_pan = tool
            return
        if self._active_tool is tool:
            return
        if self._active_tool is not None and self._document is not None:
            self._active_tool.on_deactivate(self._document)
        self._active_tool = tool
        if self._active_tool is not None and self._document is not None:
            self._active_tool.on_activate(self._document)
        self._update_cursor()
        self.update()

    def set_tool(self, tool: Optional[BaseTool]) -> None:
        """Alias for set_active_tool."""
        self.set_active_tool(tool)

    def set_hand_tool(self, hand_tool: Optional[BaseTool]) -> None:
        self._hand_tool = hand_tool

    @property
    def sprite_preview_boxes(self) -> list[ComponentBox]:
        return self._sprite_preview_boxes

    def set_sprite_preview_boxes(self, boxes: Optional[list[ComponentBox]]) -> None:
        self._sprite_preview_boxes = list(boxes) if boxes else []
        self.update()

    @property
    def is_temp_panning(self) -> bool:
        return self._temp_pan_active or self._space_pressed

    def fit_to_view(self) -> None:
        if self._document is None:
            return
        w = max(1, self.width())
        h = max(1, self.height())
        self._document.viewport.reset_view(w, h, self._document.width, self._document.height)
        self.zoomChanged.emit(self._document.viewport.zoom)
        self.update()

    def set_zoom_100(self) -> None:
        if self._document is None:
            return
        vp = self._document.viewport
        cx_screen = self.width() * 0.5
        cy_screen = self.height() * 0.5
        factor = 1.0 / max(vp.zoom, 1e-6)
        new_zoom = vp.zoom_at_cursor(cx_screen, cy_screen, factor)
        self.zoomChanged.emit(new_zoom)
        self.update()

    def _invalidate_selection_cache(self) -> None:
        self._sel_cached_ref = None
        self._sel_bbox = None
        self._sel_is_rect = True
        self._sel_lines = []
        self._sel_segments_np: Optional[np.ndarray] = None

    def _sync_march_timer(self) -> None:
        has_sel = self._document is not None and self._document.selection_mask is not None
        if has_sel:
            if not self._march_timer.isActive():
                self._march_timer.start()
        else:
            if self._march_timer.isActive():
                self._march_timer.stop()

    def _advance_march(self) -> None:
        self._march_offset = (self._march_offset + 1) % 8
        if self._document is not None and self._document.selection_mask is not None:
            self.update()
        else:
            self._march_timer.stop()

    def _get_checker_brush(self) -> QBrush:
        if self._checker_brush is None:
            self._checker_brush = _create_checkerboard_brush(8)
        return self._checker_brush

    def _update_cursor(self) -> None:
        if self._temp_pan_active:
            self.setCursor(Qt.ClosedHandCursor)
            return
        if self._space_pressed:
            self.setCursor(Qt.OpenHandCursor)
            return
        if self._active_tool is not None:
            hint = getattr(self._active_tool, "cursor_hint", "crosshair")
            shape = _CURSOR_MAP.get(str(hint).lower(), Qt.CrossCursor)
            self.setCursor(shape)
        else:
            self.setCursor(Qt.ArrowCursor)

    def _begin_temp_pan_tool(self) -> None:
        if self._saved_tool_before_pan is None and self._active_tool is not self._hand_tool:
            self._saved_tool_before_pan = self._active_tool
        if self._hand_tool is not None:
            self._active_tool = self._hand_tool

    def _end_temp_pan(self, force_restore: bool = False) -> None:
        if force_restore:
            self._space_pressed = False
            self._temp_pan_active = False
            self._temp_pan_button = None
            self._last_pan_pos = None
        if not self._space_pressed and not self._temp_pan_active:
            if self._saved_tool_before_pan is not None:
                self._active_tool = self._saved_tool_before_pan
                self._saved_tool_before_pan = None
        self._update_cursor()

    @staticmethod
    def _extract_contour_segments(sel: np.ndarray, bx: int, by: int, bw: int, bh: int) -> np.ndarray:
        """Extract merged horizontal & vertical boundary segments in canvas coordinates."""
        sub = sel[by : by + bh, bx : bx + bw] > 0
        padded = np.pad(sub, 1, mode="constant", constant_values=False)

        # Horizontal boundaries along row lines y = by + r
        h_edges = padded[:-1, 1:-1] != padded[1:, 1:-1]
        h_pad = np.pad(h_edges, ((0, 0), (1, 1)), mode="constant", constant_values=False)
        h_diff = h_pad[:, 1:] != h_pad[:, :-1]
        ry, rx = np.nonzero(h_diff)
        h_segs = np.empty((ry.size // 2, 4), dtype=np.float32)
        if ry.size >= 2:
            h_segs[:, 0] = bx + rx[0::2]
            h_segs[:, 1] = by + ry[0::2]
            h_segs[:, 2] = bx + rx[1::2]
            h_segs[:, 3] = by + ry[0::2]

        # Vertical boundaries along column lines x = bx + c
        v_edges = (padded[1:-1, :-1] != padded[1:-1, 1:]).T
        v_pad = np.pad(v_edges, ((0, 0), (1, 1)), mode="constant", constant_values=False)
        v_diff = v_pad[:, 1:] != v_pad[:, :-1]
        cx_idx, cy_idx = np.nonzero(v_diff)
        v_segs = np.empty((cx_idx.size // 2, 4), dtype=np.float32)
        if cx_idx.size >= 2:
            v_segs[:, 0] = bx + cx_idx[0::2]
            v_segs[:, 1] = by + cy_idx[0::2]
            v_segs[:, 2] = bx + cx_idx[0::2]
            v_segs[:, 3] = by + cy_idx[1::2]

        if h_segs.shape[0] == 0:
            return v_segs
        if v_segs.shape[0] == 0:
            return h_segs
        return np.vstack((h_segs, v_segs))

    def _draw_selection_overlay(self, painter: QPainter, doc: DocumentSession) -> None:
        sel = doc.selection_mask
        if sel is None:
            self._invalidate_selection_cache()
            return

        vp = doc.viewport
        if self._sel_cached_ref is not sel:
            self._sel_cached_ref = sel
            self._sel_segments_np = None
            nz_rows = np.flatnonzero(np.any(sel > 0, axis=1))
            if nz_rows.size == 0:
                self._sel_bbox = None
                return
            nz_cols = np.flatnonzero(np.any(sel > 0, axis=0))
            x0, x1 = int(nz_cols[0]), int(nz_cols[-1])
            y0, y1 = int(nz_rows[0]), int(nz_rows[-1])
            bw, bh = x1 - x0 + 1, y1 - y0 + 1
            self._sel_bbox = (x0, y0, bw, bh)
            sub = sel[y0 : y1 + 1, x0 : x1 + 1] > 0
            self._sel_is_rect = bool(np.all(sub))
            if not self._sel_is_rect:
                segs = self._extract_contour_segments(sel, x0, y0, bw, bh)
                if segs.shape[0] <= 40000:
                    self._sel_segments_np = segs

        if self._sel_bbox is None:
            return

        bx, by, bw, bh = self._sel_bbox
        painter.save()
        black_pen = QPen(QColor(0, 0, 0, 225), 1.5, Qt.SolidLine)
        dash_pen = QPen(QColor(255, 255, 255, 245), 1.25, Qt.CustomDashLine)
        dash_pen.setDashPattern([4.0, 4.0])
        dash_pen.setDashOffset(float(self._march_offset))

        if self._sel_is_rect or self._sel_segments_np is None:
            sx0, sy0 = vp.canvas_to_screen(float(bx), float(by))
            sx1, sy1 = vp.canvas_to_screen(float(bx + bw), float(by + bh))
            rect = QRectF(sx0, sy0, sx1 - sx0, sy1 - sy0)
            painter.setPen(black_pen)
            painter.drawRect(rect)
            painter.setPen(dash_pen)
            painter.drawRect(rect)
            painter.restore()
            return

        segs = self._sel_segments_np
        if segs.shape[0] > 0:
            sc = np.empty_like(segs)
            sc[:, 0] = segs[:, 0] * vp.zoom + vp.pan_x
            sc[:, 1] = segs[:, 1] * vp.zoom + vp.pan_y
            sc[:, 2] = segs[:, 2] * vp.zoom + vp.pan_x
            sc[:, 3] = segs[:, 3] * vp.zoom + vp.pan_y
            q_lines = [QLineF(a, b, c, d) for a, b, c, d in sc.tolist()]
            painter.setPen(black_pen)
            painter.drawLines(q_lines)
            painter.setPen(dash_pen)
            painter.drawLines(q_lines)
        painter.restore()

    def _draw_snap_guides(self, painter: QPainter, doc: DocumentSession) -> None:
        if self._tool_context is None or not self._tool_context.snap_guides:
            return
        vp = doc.viewport
        center_x = doc.width * 0.5
        center_y = doc.height * 0.5

        painter.save()
        for axis, coord in self._tool_context.snap_guides:
            if axis == "vertical":
                is_center = abs(float(coord) - center_x) < 1.0
                color = QColor(255, 0, 220, 230) if is_center else QColor(0, 229, 255, 230)
                sx, _ = vp.canvas_to_screen(float(coord), 0.0)
                pen = QPen(color, 1.0, Qt.SolidLine)
                pen.setCosmetic(True)
                painter.setPen(pen)
                painter.drawLine(QPointF(sx, 0.0), QPointF(sx, float(self.height())))
            elif axis == "horizontal":
                is_center = abs(float(coord) - center_y) < 1.0
                color = QColor(255, 0, 220, 230) if is_center else QColor(0, 229, 255, 230)
                _, sy = vp.canvas_to_screen(0.0, float(coord))
                pen = QPen(color, 1.0, Qt.SolidLine)
                pen.setCosmetic(True)
                painter.setPen(pen)
                painter.drawLine(QPointF(0.0, sy), QPointF(float(self.width()), sy))
        painter.restore()

    def _draw_sprite_boxes(self, painter: QPainter, doc: DocumentSession) -> None:
        if not self._sprite_preview_boxes:
            return
        vp = doc.viewport
        painter.save()
        border_pen = QPen(QColor(0, 229, 255, 240), 1.5, Qt.SolidLine)
        border_pen.setCosmetic(True)
        fill_brush = QBrush(QColor(0, 229, 255, 30))
        badge_font = QFont("Segoe UI", 9, QFont.Bold)
        painter.setFont(badge_font)

        for idx, box in enumerate(self._sprite_preview_boxes, start=1):
            sx0, sy0 = vp.canvas_to_screen(float(box.x), float(box.y))
            sx1, sy1 = vp.canvas_to_screen(
                float(box.x + box.width), float(box.y + box.height)
            )
            rect = QRectF(sx0, sy0, sx1 - sx0, sy1 - sy0)
            painter.fillRect(rect, fill_brush)
            painter.setPen(border_pen)
            painter.drawRect(rect)

            label = f"#{idx}"
            badge_w = max(22.0, float(len(label) * 8 + 8))
            badge_rect = QRectF(sx0, max(0.0, sy0 - 16.0), badge_w, 16.0)
            painter.fillRect(badge_rect, QColor(0, 150, 190, 230))
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(badge_rect, Qt.AlignCenter, label)
        painter.restore()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(24, 24, 24))

        doc = self._document
        if doc is None:
            painter.setPen(QColor(110, 110, 115))
            painter.drawText(self.rect(), Qt.AlignCenter, tr("canvas.empty_hint"))
            painter.end()
            return

        vp = doc.viewport
        sx0, sy0 = vp.canvas_to_screen(0.0, 0.0)
        sx1, sy1 = vp.canvas_to_screen(float(doc.width), float(doc.height))
        canvas_rect = QRectF(sx0, sy0, sx1 - sx0, sy1 - sy0)

        # Checkerboard transparency backdrop clipped to canvas area
        painter.save()
        painter.setClipRect(canvas_rect)
        painter.fillRect(canvas_rect, self._get_checker_brush())
        painter.restore()

        # Zero-copy composite QImage view
        buf = doc.composite()
        qimg = QImage(buf.data, doc.width, doc.height, doc.width * 4, QImage.Format_RGBA8888)

        painter.save()
        painter.translate(vp.pan_x, vp.pan_y)
        painter.scale(vp.zoom, vp.zoom)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, vp.zoom < 1.0)
        painter.drawImage(0, 0, qimg)
        painter.restore()

        # Subtle canvas boundary frame
        painter.save()
        painter.setPen(QPen(QColor(68, 68, 74), 1.0))
        painter.drawRect(canvas_rect)
        painter.restore()

        # Selection marching-ants outline
        self._draw_selection_overlay(painter, doc)

        # Magnetic snap guides
        self._draw_snap_guides(painter, doc)

        # Smart Alpha Splitter preview boxes
        self._draw_sprite_boxes(painter, doc)

        # Active tool overlay
        if self._active_tool is not None and not self._temp_pan_active:
            painter.save()
            self._active_tool.draw_overlay(painter, doc, vp)
            painter.restore()

        painter.end()

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self._document is None:
            event.ignore()
            return
        delta = event.angleDelta().y()
        if delta == 0:
            delta = event.pixelDelta().y()
        if delta != 0:
            factor = 1.15 ** (delta / 120.0)
            pos = event.position()
            new_zoom = self._document.viewport.zoom_at_cursor(
                float(pos.x()), float(pos.y()), factor
            )
            self.zoomChanged.emit(new_zoom)
            self.update()
        event.accept()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self.setFocus()
        if self._document is None:
            super().mousePressEvent(event)
            return

        pos = event.position()
        sx, sy = float(pos.x()), float(pos.y())
        cx, cy = self._document.viewport.screen_to_canvas(sx, sy)
        self.cursorMoved.emit(cx, cy)

        # Temporary Hand panning: Space + Left Drag OR Middle Mouse Drag
        if event.button() == Qt.MiddleButton or (
            self._space_pressed and event.button() == Qt.LeftButton
        ):
            self._begin_temp_pan_tool()
            self._temp_pan_active = True
            self._temp_pan_button = event.button()
            self._last_pan_pos = pos
            self._update_cursor()
            event.accept()
            return

        if self._space_pressed:
            event.accept()
            return

        if self._active_tool is not None:
            mods = extract_modifiers(event.modifiers())
            btn = _qt_flag_int(event.button())
            self._active_tool.mouse_press(
                self._document, cx, cy, button=btn, modifiers=mods, sx=sx, sy=sy
            )
            self._invalidate_selection_cache()
            self._sync_march_timer()
            self.update()
            self.documentModified.emit()
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._document is None:
            super().mouseMoveEvent(event)
            return

        pos = event.position()
        sx, sy = float(pos.x()), float(pos.y())
        cx, cy = self._document.viewport.screen_to_canvas(sx, sy)
        self.cursorMoved.emit(cx, cy)

        if self._temp_pan_active and self._last_pan_pos is not None:
            dx = sx - float(self._last_pan_pos.x())
            dy = sy - float(self._last_pan_pos.y())
            self._document.viewport.pan(dx, dy)
            self._last_pan_pos = pos
            self.update()
            event.accept()
            return

        if self._space_pressed:
            event.accept()
            return

        if self._active_tool is not None:
            mods = extract_modifiers(event.modifiers())
            buttons = _qt_flag_int(event.buttons())
            self._active_tool.mouse_move(
                self._document, cx, cy, buttons=buttons, modifiers=mods, sx=sx, sy=sy
            )
            self.update()
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._document is None:
            super().mouseReleaseEvent(event)
            return

        pos = event.position()
        sx, sy = float(pos.x()), float(pos.y())
        cx, cy = self._document.viewport.screen_to_canvas(sx, sy)

        if self._temp_pan_active and event.button() == self._temp_pan_button:
            self._temp_pan_active = False
            self._temp_pan_button = None
            self._last_pan_pos = None
            self._end_temp_pan()
            self.update()
            event.accept()
            return

        if self._space_pressed:
            event.accept()
            return

        if self._active_tool is not None:
            mods = extract_modifiers(event.modifiers())
            btn = _qt_flag_int(event.button())
            self._active_tool.mouse_release(
                self._document, cx, cy, button=btn, modifiers=mods, sx=sx, sy=sy
            )
            self._invalidate_selection_cache()
            self._sync_march_timer()
            self.update()
            self.documentModified.emit()
            event.accept()
            return

        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            if not self._space_pressed:
                self._space_pressed = True
                self._begin_temp_pan_tool()
                self._update_cursor()
                self.update()
            event.accept()
            return

        if self._document is not None and self._active_tool is not None:
            mods = extract_modifiers(event.modifiers())
            handled = self._active_tool.key_press(self._document, event.key(), mods)
            if handled:
                self._invalidate_selection_cache()
                self._sync_march_timer()
                self.update()
                self.documentModified.emit()
                event.accept()
                return

        if event.key() == Qt.Key_Escape and self._document is not None:
            if self._document.selection_mask is not None:
                self._document.clear_selection()
                self._invalidate_selection_cache()
                self._sync_march_timer()
                self.update()
                event.accept()
                return

        super().keyPressEvent(event)

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
            self.filesDropped.emit(paths)
            event.acceptProposedAction()
            return
        super().dropEvent(event)

    def contextMenuEvent(self, event: QContextMenuEvent) -> None:
        if not self._temp_pan_active:
            self.contextMenuRequested.emit(event.globalPos())
            event.accept()
            return
        super().contextMenuEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self._space_pressed = False
            self._end_temp_pan()
            self.update()
            event.accept()
            return
        super().keyReleaseEvent(event)

    def closeEvent(self, event) -> None:
        self._march_timer.stop()
        super().closeEvent(event)
