"""Utility tools: Eyedropper, Bucket Fill, Linear Gradient, and Text rasterization."""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from app.core.c_bridge import get_engine
from app.core.document import compute_dirty_rect
from app.tools.base_tool import (
    MOD_ALT,
    BaseTool,
    ToolContext,
    ToolType,
    align_selection_mask_to_layer,
    extract_button,
    extract_modifiers,
)

if TYPE_CHECKING:
    from app.core.document import DocumentSession, ViewportTransform

_HEADLESS_QAPP: Any = None


def _ensure_qgui_app() -> None:
    """Create a QApplication (or fallback QGuiApplication) if none exists (needed by QFont)."""
    global _HEADLESS_QAPP
    from PySide6.QtGui import QGuiApplication

    if QGuiApplication.instance() is not None:
        return

    try:
        from PySide6.QtWidgets import QApplication

        _HEADLESS_QAPP = QApplication([])
    except ImportError:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        _HEADLESS_QAPP = QGuiApplication([])


def pick_color_at(
    doc: "DocumentSession",
    context: ToolContext,
    cx: float,
    cy: float,
) -> Optional[tuple[int, int, int, int]]:
    """Sample composited RGBA pixel at (cx, cy), update context.fg_color, and notify."""
    ix = int(math.floor(cx))
    iy = int(math.floor(cy))
    if not (0 <= ix < doc.width and 0 <= iy < doc.height):
        return None
    comp = doc.composite()
    px = comp[iy, ix]
    color = (int(px[0]), int(px[1]), int(px[2]), int(px[3]))
    context.fg_color = color
    context.notify_color_picked(color)
    return color


def _composite_source_over_patch(
    dst_patch: np.ndarray,
    src_patch: np.ndarray,
    sel_patch: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Blend straight RGBA src_patch over dst_patch (uint8 HxWx4) with optional mask."""
    sa = src_patch[:, :, 3].astype(np.float32) * (1.0 / 255.0)
    if sel_patch is not None:
        sa *= sel_patch.astype(np.float32) * (1.0 / 255.0)
    da = dst_patch[:, :, 3].astype(np.float32) * (1.0 / 255.0)
    out_a = sa + da * (1.0 - sa)

    out = np.zeros_like(dst_patch)
    nz = out_a > 1e-6
    if not np.any(nz):
        return out

    sr = src_patch[:, :, :3].astype(np.float32)
    dr = dst_patch[:, :, :3].astype(np.float32)
    sa_e = sa[:, :, None]
    da_e = da[:, :, None]
    k_dst = da_e * (1.0 - sa_e)
    inv_oa = np.zeros_like(out_a)
    inv_oa[nz] = 1.0 / out_a[nz]

    rgb = (sr * sa_e + dr * k_dst) * inv_oa[:, :, None]
    out[:, :, :3] = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    out[:, :, 3] = np.clip(np.rint(out_a * 255.0), 0, 255).astype(np.uint8)
    return out


class EyedropperTool(BaseTool):
    """Samples composited pixel colors into ToolContext.fg_color."""

    tool_id = ToolType.EYEDROPPER.value
    name = "Eyedropper"
    cursor_hint = "cross"

    def __init__(self, context: ToolContext) -> None:
        super().__init__(context)
        self._dragging = False

    def mouse_press(
        self,
        doc: "DocumentSession",
        cx: float,
        cy: float,
        button: int = 1,
        modifiers: int = 0,
        sx: float = 0.0,
        sy: float = 0.0,
    ) -> None:
        if extract_button(button) != 1:
            return
        self._dragging = True
        pick_color_at(doc, self.context, cx, cy)

    def mouse_move(
        self,
        doc: "DocumentSession",
        cx: float,
        cy: float,
        buttons: int = 0,
        modifiers: int = 0,
        sx: float = 0.0,
        sy: float = 0.0,
    ) -> None:
        btn = extract_button(buttons)
        if not self._dragging or (btn != 0 and (btn & 1) == 0):
            return
        pick_color_at(doc, self.context, cx, cy)

    def mouse_release(
        self,
        doc: "DocumentSession",
        cx: float,
        cy: float,
        button: int = 1,
        modifiers: int = 0,
        sx: float = 0.0,
        sy: float = 0.0,
    ) -> None:
        self._dragging = False

    def on_deactivate(self, doc: Optional["DocumentSession"] = None) -> None:
        self._dragging = False


class BucketFillTool(BaseTool):
    """C-accelerated flood fill with Alt-click color picking and Delta Undo."""

    tool_id = ToolType.BUCKET_FILL.value
    name = "Bucket Fill"
    cursor_hint = "cross"

    def mouse_press(
        self,
        doc: "DocumentSession",
        cx: float,
        cy: float,
        button: int = 1,
        modifiers: int = 0,
        sx: float = 0.0,
        sy: float = 0.0,
    ) -> None:
        if extract_button(button) != 1:
            return
        if extract_modifiers(modifiers) & MOD_ALT:
            pick_color_at(doc, self.context, cx, cy)
            return

        layer = doc.active_layer
        if layer is None or layer.locked:
            return

        lx = int(math.floor(cx)) - layer.offset_x
        ly = int(math.floor(cy)) - layer.offset_y
        if not (0 <= lx < layer.width and 0 <= ly < layer.height):
            return

        sel = align_selection_mask_to_layer(doc, layer)
        before_snapshot = layer.pixels.copy()
        filled, dirty = get_engine().flood_fill(
            layer.pixels,
            seed_x=lx,
            seed_y=ly,
            fill_rgba=self.context.fg_color,
            tolerance=int(self.context.fill_tolerance),
            contiguous=bool(self.context.fill_contiguous),
            selection_mask=sel,
        )
        if filled <= 0 or dirty is None:
            return

        dx, dy, dw, dh = dirty
        before_roi = before_snapshot[dy : dy + dh, dx : dx + dw]
        after_roi = layer.pixels[dy : dy + dh, dx : dx + dw]
        rel = compute_dirty_rect(before_roi, after_roi)
        if rel is None:
            return

        rx, ry, rw, rh = rel
        abs_rect = (dx + rx, dy + ry, rw, rh)
        ax, ay = abs_rect[0], abs_rect[1]
        doc.record_delta_patch(
            layer_id=layer.id,
            dirty_rect=abs_rect,
            before_patch=before_snapshot[ay : ay + rh, ax : ax + rw],
            after_patch=layer.pixels[ay : ay + rh, ax : ax + rw],
            target="pixels",
            description="Bucket Fill",
        )
        self.context.notify_canvas_invalidated()


class GradientTool(BaseTool):
    """Linear RGBA gradient from fg_color to bg_color respecting selection_mask."""

    tool_id = ToolType.GRADIENT.value
    name = "Gradient"
    cursor_hint = "cross"

    def __init__(self, context: ToolContext) -> None:
        super().__init__(context)
        self._dragging = False
        self._start: tuple[float, float] = (0.0, 0.0)
        self._end: tuple[float, float] = (0.0, 0.0)

    def mouse_press(
        self,
        doc: "DocumentSession",
        cx: float,
        cy: float,
        button: int = 1,
        modifiers: int = 0,
        sx: float = 0.0,
        sy: float = 0.0,
    ) -> None:
        if extract_button(button) != 1:
            return
        if extract_modifiers(modifiers) & MOD_ALT:
            pick_color_at(doc, self.context, cx, cy)
            return

        layer = doc.active_layer
        if layer is None or layer.locked:
            return

        self._dragging = True
        self._start = (float(cx), float(cy))
        self._end = (float(cx), float(cy))
        self.context.notify_canvas_invalidated()

    def mouse_move(
        self,
        doc: "DocumentSession",
        cx: float,
        cy: float,
        buttons: int = 0,
        modifiers: int = 0,
        sx: float = 0.0,
        sy: float = 0.0,
    ) -> None:
        btn = extract_button(buttons)
        if not self._dragging or (btn != 0 and (btn & 1) == 0):
            return
        self._end = (float(cx), float(cy))
        self.context.notify_canvas_invalidated()

    def mouse_release(
        self,
        doc: "DocumentSession",
        cx: float,
        cy: float,
        button: int = 1,
        modifiers: int = 0,
        sx: float = 0.0,
        sy: float = 0.0,
    ) -> None:
        if not self._dragging:
            return
        self._dragging = False
        self._end = (float(cx), float(cy))

        layer = doc.active_layer
        if layer is None or layer.locked:
            self.context.notify_canvas_invalidated()
            return

        lx0 = self._start[0] - layer.offset_x
        ly0 = self._start[1] - layer.offset_y
        lx1 = self._end[0] - layer.offset_x
        ly1 = self._end[1] - layer.offset_y
        dx = lx1 - lx0
        dy = ly1 - ly0
        len2 = dx * dx + dy * dy
        if len2 < 1e-6:
            self.context.notify_canvas_invalidated()
            return

        sel = align_selection_mask_to_layer(doc, layer)
        if sel is not None:
            ys, xs = np.nonzero(sel)
            if ys.size == 0:
                self.context.notify_canvas_invalidated()
                return
            rx0, ry0 = int(xs.min()), int(ys.min())
            rx1, ry1 = int(xs.max()) + 1, int(ys.max()) + 1
            sel_patch = sel[ry0:ry1, rx0:rx1]
        else:
            rx0, ry0, rx1, ry1 = 0, 0, layer.width, layer.height
            sel_patch = None

        yy = np.arange(ry0, ry1, dtype=np.float32)[:, None]
        xx = np.arange(rx0, rx1, dtype=np.float32)[None, :]
        t = np.clip(((xx - lx0) * dx + (yy - ly0) * dy) / float(len2), 0.0, 1.0)

        fg = np.array(self.context.fg_color, dtype=np.float32)
        bg = np.array(self.context.bg_color, dtype=np.float32)
        fg_a = fg[3] / 255.0
        bg_a = bg[3] / 255.0

        # Interpolate in premultiplied RGB to avoid dark fringes when fading to transparent
        fg_pm = fg[:3] * fg_a
        bg_pm = bg[:3] * bg_a
        out_a = fg_a * (1.0 - t) + bg_a * t
        out_pm = fg_pm[None, None, :] * (1.0 - t[:, :, None]) + bg_pm[None, None, :] * t[:, :, None]

        grad_patch = np.zeros((ry1 - ry0, rx1 - rx0, 4), dtype=np.uint8)
        nz = out_a > 1e-6
        if np.any(nz):
            rgb = np.zeros((ry1 - ry0, rx1 - rx0, 3), dtype=np.float32)
            rgb[nz] = out_pm[nz] / out_a[nz][:, None]
            grad_patch[:, :, :3] = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
            grad_patch[:, :, 3] = np.clip(np.rint(out_a * 255.0), 0, 255).astype(np.uint8)

        before_patch = layer.pixels[ry0:ry1, rx0:rx1].copy()
        after_patch = _composite_source_over_patch(before_patch, grad_patch, sel_patch=sel_patch)

        rel = compute_dirty_rect(before_patch, after_patch)
        if rel is not None:
            layer.pixels[ry0:ry1, rx0:rx1] = after_patch
            rx, ry, rw, rh = rel
            abs_rect = (rx0 + rx, ry0 + ry, rw, rh)
            doc.record_delta_patch(
                layer_id=layer.id,
                dirty_rect=abs_rect,
                before_patch=before_patch[ry : ry + rh, rx : rx + rw],
                after_patch=after_patch[ry : ry + rh, rx : rx + rw],
                target="pixels",
                description="Linear Gradient",
            )
        self.context.notify_canvas_invalidated()

    def draw_overlay(
        self,
        painter: Any,
        doc: "DocumentSession",
        viewport: "ViewportTransform",
    ) -> None:
        if not self._dragging or painter is None:
            return
        from PySide6.QtCore import QPointF, Qt
        from PySide6.QtGui import QColor, QPen

        sx0, sy0 = viewport.canvas_to_screen(*self._start)
        sx1, sy1 = viewport.canvas_to_screen(*self._end)
        painter.save()
        painter.setPen(QPen(QColor(0, 0, 0, 200), 3.0))
        painter.drawLine(QPointF(sx0, sy0), QPointF(sx1, sy1))
        painter.setPen(QPen(QColor(255, 255, 255, 240), 1.5, Qt.PenStyle.DashLine))
        painter.drawLine(QPointF(sx0, sy0), QPointF(sx1, sy1))
        painter.setBrush(QColor(*self.context.fg_color[:3], 255))
        painter.setPen(QPen(QColor(255, 255, 255), 1.5))
        painter.drawEllipse(QPointF(sx0, sy0), 4.5, 4.5)
        painter.setBrush(QColor(*self.context.bg_color[:3], 255))
        painter.drawEllipse(QPointF(sx1, sy1), 4.5, 4.5)
        painter.restore()

    def on_deactivate(self, doc: Optional["DocumentSession"] = None) -> None:
        self._dragging = False


class TextTool(BaseTool):
    """Rasterizes context.text_content onto the active layer with Delta Undo."""

    tool_id = ToolType.TEXT.value
    name = "Text"
    cursor_hint = "ibeam"

    def mouse_press(
        self,
        doc: "DocumentSession",
        cx: float,
        cy: float,
        button: int = 1,
        modifiers: int = 0,
        sx: float = 0.0,
        sy: float = 0.0,
    ) -> None:
        if extract_button(button) != 1:
            return
        if extract_modifiers(modifiers) & MOD_ALT:
            pick_color_at(doc, self.context, cx, cy)
            return

        layer = doc.active_layer
        if layer is None or layer.locked:
            return
        text = self.context.text_content
        if not text:
            return

        _ensure_qgui_app()
        from PySide6.QtCore import QRect, Qt
        from PySide6.QtGui import QColor, QFont, QFontMetrics, QImage, QPainter

        font = QFont(self.context.font_family, max(1, int(self.context.font_size)))
        font.setBold(bool(self.context.font_bold))
        fm = QFontMetrics(font)
        flags = int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        bounds = fm.boundingRect(QRect(0, 0, 4096, 4096), flags, text)

        pad = 6
        pw = max(1, bounds.width() + pad * 2)
        ph = max(1, bounds.height() + pad * 2)

        qimg = QImage(pw, ph, QImage.Format.Format_RGBA8888)
        qimg.fill(0)
        painter = QPainter(qimg)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        painter.setFont(font)
        r, g, b, a = self.context.fg_color
        painter.setPen(QColor(r & 0xFF, g & 0xFF, b & 0xFF, a & 0xFF))
        painter.drawText(QRect(pad, pad, pw - pad, ph - pad), flags, text)
        painter.end()

        stride_px = qimg.bytesPerLine() // 4
        raw = np.frombuffer(
            qimg.constBits(),
            dtype=np.uint8,
            count=qimg.height() * qimg.bytesPerLine(),
        ).reshape((qimg.height(), stride_px, 4))[:, :pw, :4].copy()

        lx = int(round(cx)) - layer.offset_x - pad
        ly = int(round(cy)) - layer.offset_y - pad

        x0 = max(0, lx)
        y0 = max(0, ly)
        x1 = min(layer.width, lx + pw)
        y1 = min(layer.height, ly + ph)
        if x1 <= x0 or y1 <= y0:
            return

        sx0 = x0 - lx
        sy0 = y0 - ly
        sx1 = sx0 + (x1 - x0)
        sy1 = sy0 + (y1 - y0)

        src_patch = raw[sy0:sy1, sx0:sx1]
        sel = align_selection_mask_to_layer(doc, layer)
        sel_patch = sel[y0:y1, x0:x1] if sel is not None else None

        before_patch = layer.pixels[y0:y1, x0:x1].copy()
        after_patch = _composite_source_over_patch(before_patch, src_patch, sel_patch=sel_patch)

        rel = compute_dirty_rect(before_patch, after_patch)
        if rel is None:
            return

        layer.pixels[y0:y1, x0:x1] = after_patch
        rx, ry, rw, rh = rel
        abs_rect = (x0 + rx, y0 + ry, rw, rh)
        doc.record_delta_patch(
            layer_id=layer.id,
            dirty_rect=abs_rect,
            before_patch=before_patch[ry : ry + rh, rx : rx + rw],
            after_patch=after_patch[ry : ry + rh, rx : rx + rw],
            target="pixels",
            description="Text",
        )
        self.context.notify_canvas_invalidated()
