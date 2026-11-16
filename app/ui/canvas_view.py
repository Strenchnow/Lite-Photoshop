"""High-FPS QWidget viewport for rendering DocumentSession, overlays, zoom-to-cursor, and panning."""

from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
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

from app.core.c_bridge import ComponentBox
from app.core.document import DocumentSession
from app.tools.base_tool import BaseTool, ToolContext, extract_modifiers


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

        # Selection outline animation & cache
        self._march_offset: int = 0
        self._march_timer = QTimer(self)
        self._march_timer.setInterval(160)
        self._march_timer.timeout.connect(self._advance_march)
        self._sel_cache_id: Optional[int] = None
        self._sel_cache_phase: int = -1
        self._sel_bbox: Optional[tuple[int, int, int, int]] = None
        self._sel_is_rect: bool = True
        self._sel_edge_img: Optional[QImage] = None
        self._sel_edge_buf: Optional[np.ndarray] = None

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
        self._sel_cache_id = None
        self._sel_cache_phase = -1
        self._sel_bbox = None
        self._sel_edge_img = None
        self._sel_edge_buf = None

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

    def _draw_selection_overlay(self, painter: QPainter, doc: DocumentSession) -> None:
        sel = doc.selection_mask
        if sel is None:
            self._invalidate_selection_cache()
            return

        vp = doc.viewport
        sel_id = id(sel)
        if self._sel_cache_id != sel_id:
            self._sel_cache_id = sel_id
            self._sel_cache_phase = -1
            nz_rows = np.flatnonzero(np.any(sel > 0, axis=1))
            if nz_rows.size == 0:
                self._sel_bbox = None
                self._sel_edge_img = None
                self._sel_edge_buf = None
                return
            nz_cols = np.flatnonzero(np.any(sel > 0, axis=0))
            x0, x1 = int(nz_cols[0]), int(nz_cols[-1])
            y0, y1 = int(nz_rows[0]), int(nz_rows[-1])
            bw, bh = x1 - x0 + 1, y1 - y0 + 1
            self._sel_bbox = (x0, y0, bw, bh)
            sub = sel[y0 : y1 + 1, x0 : x1 + 1] > 0
            self._sel_is_rect = bool(np.all(sub))
            self._sel_edge_img = None
            self._sel_edge_buf = None

        if self._sel_bbox is None:
            return

        bx, by, bw, bh = self._sel_bbox
        if self._sel_is_rect or (bw * bh > 2048 * 2048):
            sx0, sy0 = vp.canvas_to_screen(float(bx), float(by))
            sx1, sy1 = vp.canvas_to_screen(float(bx + bw), float(by + bh))
            rect = QRectF(sx0, sy0, sx1 - sx0, sy1 - sy0)
            painter.save()
            painter.setPen(QPen(QColor(0, 0, 0, 220), 1.0, Qt.SolidLine))
            painter.drawRect(rect)
            dash_pen = QPen(QColor(255, 255, 255, 240), 1.0, Qt.CustomDashLine)
            dash_pen.setDashPattern([4.0, 4.0])
            dash_pen.setDashOffset(float(self._march_offset))
            painter.setPen(dash_pen)
            painter.drawRect(rect)
            painter.restore()
            return

        # Non-rectangular selection contour within the tight bounding box
        if self._sel_cache_phase != self._march_offset or self._sel_edge_img is None:
            self._sel_cache_phase = self._march_offset
            sub = sel[by : by + bh, bx : bx + bw] > 0
            padded = np.pad(sub, 1, mode="constant", constant_values=False)
            interior = (
                padded[:-2, 1:-1]
                & padded[2:, 1:-1]
                & padded[1:-1, :-2]
                & padded[1:-1, 2:]
            )
            edges = sub & (~interior)
            buf = np.zeros((bh, bw, 4), dtype=np.uint8)
            yy, xx = np.nonzero(edges)
            if yy.size > 0:
                stripe = ((yy + xx + self._march_offset) // 4) % 2 == 0
                buf[yy[stripe], xx[stripe]] = [255, 255, 255, 245]
                buf[yy[~stripe], xx[~stripe]] = [15, 15, 15, 235]
            self._sel_edge_buf = buf
            self._sel_edge_img = QImage(
                buf.data, bw, bh, bw * 4, QImage.Format_RGBA8888
            )

        painter.save()
        painter.translate(vp.pan_x, vp.pan_y)
        painter.scale(vp.zoom, vp.zoom)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, False)
        painter.drawImage(bx, by, self._sel_edge_img)
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
            painter.drawText(self.rect(), Qt.AlignCenter, "No Document Open — File > New or Ctrl+V")
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

        super().keyPressEvent(event)

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
