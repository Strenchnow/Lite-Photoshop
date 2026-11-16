"""Selection, Crop, and Free Transform tools."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from app.core.c_bridge import get_engine
from app.core.document import CallableCommand
from app.core.layer import LayerMask
from app.tools.base_tool import (
    MOD_ALT,
    MOD_CTRL,
    MOD_SHIFT,
    BaseTool,
    ToolContext,
    ToolType,
    extract_button,
    extract_key,
    extract_modifiers,
)

if TYPE_CHECKING:
    from app.core.document import DocumentSession, ViewportTransform

_KEY_ESCAPE = 0x01000000
_KEY_RETURN = 0x01000004
_KEY_ENTER = 0x01000005
_KEY_D = 0x44


def _handle_selection_deselect_key(
    doc: "DocumentSession",
    context: ToolContext,
    key: int,
    modifiers: int,
) -> bool:
    k = extract_key(key)
    mods = extract_modifiers(modifiers)
    if k == _KEY_ESCAPE or ((mods & MOD_CTRL) and k == _KEY_D):
        if doc.selection_mask is not None:
            doc.clear_selection()
            context.notify_canvas_invalidated()
            return True
    return False


def _combine_selection(
    doc: "DocumentSession",
    new_mask: np.ndarray,
    modifiers: int,
) -> None:
    if modifiers & MOD_SHIFT:
        if doc.selection_mask is not None:
            combined = np.maximum(doc.selection_mask, new_mask)
        else:
            combined = new_mask
        doc.set_selection_mask(combined if np.any(combined) else None)
    elif modifiers & MOD_ALT:
        if doc.selection_mask is not None:
            combined = np.where(new_mask > 0, np.uint8(0), doc.selection_mask)
            doc.set_selection_mask(combined if np.any(combined) else None)
    else:
        doc.set_selection_mask(new_mask if np.any(new_mask) else None)


def _rasterize_polygon_scanline(
    width: int,
    height: int,
    points: list[tuple[float, float]],
) -> np.ndarray:
    """Rasterize a closed polygon into a uint8 (H, W) mask with values 0 or 255."""
    mask = np.zeros((height, width), dtype=np.uint8)
    n = len(points)
    if n < 3 or width <= 0 or height <= 0:
        return mask

    ys_pts = [p[1] for p in points]
    min_y = max(0, int(math.floor(min(ys_pts))))
    max_y = min(height - 1, int(math.ceil(max(ys_pts))))

    for y in range(min_y, max_y + 1):
        py = y + 0.5
        xs_inter: list[float] = []
        for i in range(n):
            ax, ay = points[i]
            bx, by = points[(i + 1) % n]
            if (ay <= py < by) or (by <= py < ay):
                ix = ax + (py - ay) * (bx - ax) / (by - ay)
                xs_inter.append(ix)
        if len(xs_inter) < 2:
            continue
        xs_inter.sort()
        for k in range(0, len(xs_inter) - 1, 2):
            x_left = xs_inter[k]
            x_right = xs_inter[k + 1]
            x0 = max(0, int(math.ceil(x_left - 0.5)))
            x1 = min(width - 1, int(math.floor(x_right - 0.5)))
            if x1 >= x0:
                mask[y, x0 : x1 + 1] = 255
    return mask


def _rasterize_polygon(
    width: int,
    height: int,
    points: list[tuple[float, float]],
) -> np.ndarray:
    """Rasterize a closed polygon into a uint8 (H, W) mask using Qt when available or scanline."""
    if len(points) < 3 or width <= 0 or height <= 0:
        return np.zeros((max(1, height), max(1, width)), dtype=np.uint8)[:height, :width]

    try:
        from PySide6.QtCore import QPointF, Qt
        from PySide6.QtGui import QColor, QGuiApplication, QImage, QPainter, QPainterPath

        if QGuiApplication.instance() is not None:
            img = QImage(width, height, QImage.Format.Format_Grayscale8)
            img.fill(0)
            path = QPainterPath()
            path.setFillRule(Qt.FillRule.OddEvenFill)
            path.moveTo(QPointF(points[0][0], points[0][1]))
            for px, py in points[1:]:
                path.lineTo(QPointF(px, py))
            path.closeSubpath()

            painter = QPainter(img)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
            painter.fillPath(path, QColor(255, 255, 255))
            painter.end()

            bpl = img.bytesPerLine()
            raw = np.frombuffer(img.constBits(), dtype=np.uint8, count=height * bpl).reshape((height, bpl))
            mask = np.ascontiguousarray(np.where(raw[:, :width] >= 128, np.uint8(255), np.uint8(0)))
            return mask
    except Exception:
        pass

    return _rasterize_polygon_scanline(width, height, points)


def _bilinear_resample_rotated_rgba(
    src: np.ndarray,
    box_w: float,
    box_h: float,
    center_x: float,
    center_y: float,
    angle_deg: float,
) -> tuple[int, int, int, int, np.ndarray]:
    """Resample `src` scaled to (box_w, box_h) and rotated by `angle_deg` around center."""
    src_h, src_w = src.shape[0], src.shape[1]
    rad = math.radians(angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)

    hw = max(0.5, box_w * 0.5)
    hh = max(0.5, box_h * 0.5)

    corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    rot_xs = [center_x + lx * cos_a - ly * sin_a for lx, ly in corners]
    rot_ys = [center_y + lx * sin_a + ly * cos_a for lx, ly in corners]

    out_x0 = int(math.floor(min(rot_xs)))
    out_y0 = int(math.floor(min(rot_ys)))
    out_x1 = int(math.ceil(max(rot_xs)))
    out_y1 = int(math.ceil(max(rot_ys)))
    out_w = max(1, out_x1 - out_x0)
    out_h = max(1, out_y1 - out_y0)

    yy = (np.arange(out_y0, out_y1, dtype=np.float32)[:, None] + 0.5) - float(center_y)
    xx = (np.arange(out_x0, out_x1, dtype=np.float32)[None, :] + 0.5) - float(center_x)

    # Inverse rotate into unrotated box coords centered at (0, 0)
    local_x = xx * cos_a + yy * sin_a
    local_y = -xx * sin_a + yy * cos_a

    # Map local [-hw, hw] x [-hh, hh] to source pixel center coords
    fx = (local_x + hw) * (float(src_w) / max(1e-4, float(box_w))) - 0.5
    fy = (local_y + hh) * (float(src_h) / max(1e-4, float(box_h))) - 0.5

    inside = (fx >= -0.5) & (fx <= src_w - 0.5) & (fy >= -0.5) & (fy <= src_h - 0.5)
    out = np.zeros((out_h, out_w, 4), dtype=np.uint8)
    if not np.any(inside):
        return out_x0, out_y0, out_w, out_h, out

    fx_c = np.clip(fx, 0.0, float(src_w - 1))
    fy_c = np.clip(fy, 0.0, float(src_h - 1))

    x0 = np.floor(fx_c).astype(np.int32)
    y0 = np.floor(fy_c).astype(np.int32)
    x1 = np.minimum(x0 + 1, src_w - 1)
    y1 = np.minimum(y0 + 1, src_h - 1)

    wx = (fx_c - x0).astype(np.float32)
    wy = (fy_c - y0).astype(np.float32)
    wx0 = 1.0 - wx
    wy0 = 1.0 - wy

    w00 = (wx0 * wy0) * inside
    w10 = (wx * wy0) * inside
    w01 = (wx0 * wy) * inside
    w11 = (wx * wy) * inside

    inv255 = np.float32(1.0 / 255.0)
    p00 = src[y0, x0].astype(np.float32)
    p10 = src[y0, x1].astype(np.float32)
    p01 = src[y1, x0].astype(np.float32)
    p11 = src[y1, x1].astype(np.float32)

    a00 = p00[:, :, 3] * inv255
    a10 = p10[:, :, 3] * inv255
    a01 = p01[:, :, 3] * inv255
    a11 = p11[:, :, 3] * inv255

    out_a = a00 * w00 + a10 * w10 + a01 * w01 + a11 * w11
    nz = out_a > 1e-6
    if np.any(nz):
        pm = (
            p00[:, :, :3] * (a00 * w00)[:, :, None]
            + p10[:, :, :3] * (a10 * w10)[:, :, None]
            + p01[:, :, :3] * (a01 * w01)[:, :, None]
            + p11[:, :, :3] * (a11 * w11)[:, :, None]
        ) * inv255
        rgb = np.zeros((out_h, out_w, 3), dtype=np.float32)
        rgb[nz] = pm[nz] / out_a[nz][:, None]
        out[:, :, :3] = np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)
        out[:, :, 3] = np.clip(np.rint(out_a * 255.0), 0, 255).astype(np.uint8)

    return out_x0, out_y0, out_w, out_h, out


def _bilinear_resample_rotated_mask(
    src_mask: np.ndarray,
    box_w: float,
    box_h: float,
    center_x: float,
    center_y: float,
    angle_deg: float,
    out_x0: int,
    out_y0: int,
    out_w: int,
    out_h: int,
) -> np.ndarray:
    src_h, src_w = src_mask.shape[0], src_mask.shape[1]
    rad = math.radians(angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    hw = max(0.5, box_w * 0.5)
    hh = max(0.5, box_h * 0.5)

    yy = (np.arange(out_y0, out_y0 + out_h, dtype=np.float32)[:, None] + 0.5) - float(center_y)
    xx = (np.arange(out_x0, out_x0 + out_w, dtype=np.float32)[None, :] + 0.5) - float(center_x)

    local_x = xx * cos_a + yy * sin_a
    local_y = -xx * sin_a + yy * cos_a
    fx = (local_x + hw) * (float(src_w) / max(1e-4, float(box_w))) - 0.5
    fy = (local_y + hh) * (float(src_h) / max(1e-4, float(box_h))) - 0.5

    inside = (fx >= -0.5) & (fx <= src_w - 0.5) & (fy >= -0.5) & (fy <= src_h - 0.5)
    fx_c = np.clip(fx, 0.0, float(src_w - 1))
    fy_c = np.clip(fy, 0.0, float(src_h - 1))

    x0 = np.floor(fx_c).astype(np.int32)
    y0 = np.floor(fy_c).astype(np.int32)
    x1 = np.minimum(x0 + 1, src_w - 1)
    y1 = np.minimum(y0 + 1, src_h - 1)

    wx = (fx_c - x0).astype(np.float32)
    wy = (fy_c - y0).astype(np.float32)

    val = (
        src_mask[y0, x0].astype(np.float32) * (1.0 - wx) * (1.0 - wy)
        + src_mask[y0, x1].astype(np.float32) * wx * (1.0 - wy)
        + src_mask[y1, x0].astype(np.float32) * (1.0 - wx) * wy
        + src_mask[y1, x1].astype(np.float32) * wx * wy
    ) * inside
    return np.clip(np.rint(val), 0, 255).astype(np.uint8)


class RectMarqueeTool(BaseTool):
    """Rectangular selection mask tool supporting Shift-add, Alt-subtract, and click-clear."""

    tool_id = ToolType.MARQUEE.value
    name = "Rectangular Marquee"
    cursor_hint = "cross"

    def __init__(self, context: ToolContext) -> None:
        super().__init__(context)
        self._dragging = False
        self._start: tuple[float, float] = (0.0, 0.0)
        self._end: tuple[float, float] = (0.0, 0.0)
        self._press_mods = 0

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
        self._start = (float(cx), float(cy))
        self._end = (float(cx), float(cy))
        self._press_mods = extract_modifiers(modifiers)
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
        mods = self._press_mods | extract_modifiers(modifiers)

        sx0, sy0 = self._start
        ex0, ey0 = self._end
        if abs(ex0 - sx0) < 1.0 and abs(ey0 - sy0) < 1.0:
            if not (mods & (MOD_SHIFT | MOD_ALT)):
                doc.clear_selection()
            self.context.notify_canvas_invalidated()
            return

        x0 = max(0, min(doc.width, int(round(min(sx0, ex0)))))
        y0 = max(0, min(doc.height, int(round(min(sy0, ey0)))))
        x1 = max(0, min(doc.width, int(round(max(sx0, ex0)))))
        y1 = max(0, min(doc.height, int(round(max(sy0, ey0)))))

        if x1 > x0 and y1 > y0:
            mask = np.zeros((doc.height, doc.width), dtype=np.uint8)
            mask[y0:y1, x0:x1] = 255
            _combine_selection(doc, mask, mods)
        elif not (mods & (MOD_SHIFT | MOD_ALT)):
            doc.clear_selection()

        self.context.notify_canvas_invalidated()

    def key_press(
        self,
        doc: "DocumentSession",
        key: int,
        modifiers: int = 0,
    ) -> bool:
        return _handle_selection_deselect_key(doc, self.context, key, modifiers)

    def draw_overlay(
        self,
        painter: Any,
        doc: "DocumentSession",
        viewport: "ViewportTransform",
    ) -> None:
        if not self._dragging or painter is None:
            return
        from PySide6.QtCore import QRectF, Qt
        from PySide6.QtGui import QColor, QPen

        sx0, sy0 = viewport.canvas_to_screen(*self._start)
        sx1, sy1 = viewport.canvas_to_screen(*self._end)
        rect = QRectF(min(sx0, sx1), min(sy0, sy1), abs(sx1 - sx0), abs(sy1 - sy0))
        painter.save()
        painter.setPen(QPen(QColor(0, 0, 0, 220), 1.5))
        painter.drawRect(rect)
        painter.setPen(QPen(QColor(255, 255, 255, 240), 1.0, Qt.PenStyle.DashLine))
        painter.drawRect(rect)
        painter.restore()

    def on_deactivate(self, doc: Optional["DocumentSession"] = None) -> None:
        self._dragging = False


class LassoTool(BaseTool):
    """Freehand polygon selection tool rasterized into doc.selection_mask."""

    tool_id = ToolType.LASSO.value
    name = "Lasso"
    cursor_hint = "cross"

    def __init__(self, context: ToolContext) -> None:
        super().__init__(context)
        self._dragging = False
        self._points: list[tuple[float, float]] = []
        self._press_mods = 0

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
        self._points = [(float(cx), float(cy))]
        self._press_mods = extract_modifiers(modifiers)
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
        pt = (float(cx), float(cy))
        if not self._points or math.hypot(pt[0] - self._points[-1][0], pt[1] - self._points[-1][1]) >= 0.5:
            self._points.append(pt)
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
        pt = (float(cx), float(cy))
        if not self._points or math.hypot(pt[0] - self._points[-1][0], pt[1] - self._points[-1][1]) >= 0.25:
            self._points.append(pt)

        mods = self._press_mods | extract_modifiers(modifiers)
        pts = list(self._points)
        self._points.clear()

        if len(pts) < 3:
            if not (mods & (MOD_SHIFT | MOD_ALT)):
                doc.clear_selection()
            self.context.notify_canvas_invalidated()
            return

        poly_mask = _rasterize_polygon(doc.width, doc.height, pts)
        if np.any(poly_mask):
            _combine_selection(doc, poly_mask, mods)
        elif not (mods & (MOD_SHIFT | MOD_ALT)):
            doc.clear_selection()

        self.context.notify_canvas_invalidated()

    def key_press(
        self,
        doc: "DocumentSession",
        key: int,
        modifiers: int = 0,
    ) -> bool:
        k = extract_key(key)
        if k == _KEY_ESCAPE and self._dragging:
            self._dragging = False
            self._points.clear()
            self.context.notify_canvas_invalidated()
            return True
        return _handle_selection_deselect_key(doc, self.context, key, modifiers)

    def draw_overlay(
        self,
        painter: Any,
        doc: "DocumentSession",
        viewport: "ViewportTransform",
    ) -> None:
        if not self._dragging or not self._points or painter is None:
            return
        from PySide6.QtCore import QPointF, Qt
        from PySide6.QtGui import QColor, QPen, QPolygonF

        screen_pts = [QPointF(*viewport.canvas_to_screen(x, y)) for x, y in self._points]
        painter.save()
        if len(screen_pts) >= 2:
            poly = QPolygonF(screen_pts)
            painter.setPen(QPen(QColor(0, 0, 0, 220), 2.0))
            painter.drawPolyline(poly)
            dash_pen = QPen(QColor(255, 255, 255, 245), 1.25, Qt.PenStyle.DashLine)
            painter.setPen(dash_pen)
            painter.drawPolyline(poly)
            # Closing preview segment back to the initial point
            close_pen = QPen(QColor(0, 210, 255, 210), 1.25, Qt.PenStyle.DotLine)
            painter.setPen(close_pen)
            painter.drawLine(screen_pts[-1], screen_pts[0])

        # Highlight start anchor point
        painter.setPen(QPen(QColor(0, 0, 0, 220), 1.0))
        painter.setBrush(QColor(255, 255, 255, 240))
        painter.drawEllipse(screen_pts[0], 3.5, 3.5)
        painter.restore()

    def on_deactivate(self, doc: Optional["DocumentSession"] = None) -> None:
        self._dragging = False
        self._points.clear()


class MagicWandTool(BaseTool):
    """C-accelerated Magic Wand flood selection tool."""

    tool_id = ToolType.MAGIC_WAND.value
    name = "Magic Wand"
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
        mods = extract_modifiers(modifiers)
        ix = int(math.floor(cx))
        iy = int(math.floor(cy))
        if not (0 <= ix < doc.width and 0 <= iy < doc.height):
            if not (mods & (MOD_SHIFT | MOD_ALT)):
                doc.clear_selection()
            self.context.notify_canvas_invalidated()
            return

        engine = get_engine()
        tol = int(self.context.wand_tolerance)
        contiguous = bool(self.context.wand_contiguous)

        layer = doc.active_layer
        canvas_mask: Optional[np.ndarray] = None
        count = 0

        use_layer = False
        if layer is not None and layer.visible:
            lx = ix - layer.offset_x
            ly = iy - layer.offset_y
            if 0 <= lx < layer.width and 0 <= ly < layer.height:
                # Sample active layer directly when opaque there, or when single-layer document
                if layer.pixels[ly, lx, 3] > 0 or len(doc.layers) == 1:
                    use_layer = True
                else:
                    comp_check = doc.composite()
                    if comp_check[iy, ix, 3] == 0:
                        use_layer = True

        if use_layer and layer is not None:
            lx = ix - layer.offset_x
            ly = iy - layer.offset_y
            src_pixels = np.ascontiguousarray(layer.pixels, dtype=np.uint8)
            local_mask, count, _ = engine.magic_wand(
                src_pixels,
                seed_x=lx,
                seed_y=ly,
                tolerance=tol,
                contiguous=contiguous,
            )
            if count > 0:
                if (
                    layer.offset_x == 0
                    and layer.offset_y == 0
                    and layer.width == doc.width
                    and layer.height == doc.height
                ):
                    canvas_mask = local_mask
                else:
                    canvas_mask = np.zeros((doc.height, doc.width), dtype=np.uint8)
                    cx0 = max(0, layer.offset_x)
                    cy0 = max(0, layer.offset_y)
                    cx1 = min(doc.width, layer.offset_x + layer.width)
                    cy1 = min(doc.height, layer.offset_y + layer.height)
                    if cx1 > cx0 and cy1 > cy0:
                        sx0 = cx0 - layer.offset_x
                        sy0 = cy0 - layer.offset_y
                        canvas_mask[cy0:cy1, cx0:cx1] = local_mask[
                            sy0 : sy0 + (cy1 - cy0), sx0 : sx0 + (cx1 - cx0)
                        ]

        if canvas_mask is None:
            comp = np.ascontiguousarray(doc.composite(), dtype=np.uint8)
            canvas_mask, count, _ = engine.magic_wand(
                comp,
                seed_x=ix,
                seed_y=iy,
                tolerance=tol,
                contiguous=contiguous,
            )

        if count > 0 and canvas_mask is not None and np.any(canvas_mask):
            _combine_selection(doc, canvas_mask, mods)
        elif not (mods & (MOD_SHIFT | MOD_ALT)):
            doc.clear_selection()

        self.context.notify_canvas_invalidated()

    def key_press(
        self,
        doc: "DocumentSession",
        key: int,
        modifiers: int = 0,
    ) -> bool:
        return _handle_selection_deselect_key(doc, self.context, key, modifiers)


class CropTool(BaseTool):
    """Interactive crop rectangle tool with aspect ratio presets, handles, Enter to apply, Esc to cancel."""

    tool_id = ToolType.CROP.value
    name = "Crop"
    cursor_hint = "cross"

    def __init__(self, context: ToolContext) -> None:
        super().__init__(context)
        self.crop_rect: Optional[tuple[int, int, int, int]] = None
        self._dragging = False
        self._drag_mode: str = "create"
        self._start: tuple[float, float] = (0.0, 0.0)
        self._orig_rect: Optional[tuple[int, int, int, int]] = None

    @staticmethod
    def _constrain_from_anchor(
        doc: "DocumentSession",
        ax: float,
        ay: float,
        cx: float,
        cy: float,
        target_ratio: Optional[float],
    ) -> Optional[tuple[int, int, int, int]]:
        ax_c = max(0.0, min(float(doc.width), ax))
        ay_c = max(0.0, min(float(doc.height), ay))
        sx = 1 if cx >= ax_c else -1
        sy = 1 if cy >= ay_c else -1
        max_w = (float(doc.width) - ax_c) if sx > 0 else ax_c
        max_h = (float(doc.height) - ay_c) if sy > 0 else ay_c

        raw_w = min(max_w, abs(cx - ax_c))
        raw_h = min(max_h, abs(cy - ay_c))

        if target_ratio is None or target_ratio <= 0.0:
            w, h = raw_w, raw_h
        else:
            if raw_w / max(raw_h, 1e-6) >= target_ratio:
                w = raw_w
                h = w / target_ratio
            else:
                h = raw_h
                w = h * target_ratio
            if w > max_w:
                w = max_w
                h = w / target_ratio
            if h > max_h:
                h = max_h
                w = h * target_ratio

        cw = int(round(w))
        ch = int(round(h))
        if cw <= 0 or ch <= 0:
            return None
        x0 = int(round(ax_c)) if sx > 0 else int(round(ax_c)) - cw
        y0 = int(round(ay_c)) if sy > 0 else int(round(ay_c)) - ch
        x0 = max(0, min(doc.width - cw, x0))
        y0 = max(0, min(doc.height - ch, y0))
        return (x0, y0, cw, ch)

    def _handle_positions(self) -> dict[str, tuple[float, float]]:
        if self.crop_rect is None:
            return {}
        x, y, w, h = self.crop_rect
        mx = x + w * 0.5
        my = y + h * 0.5
        return {
            "nw": (float(x), float(y)),
            "n": (mx, float(y)),
            "ne": (float(x + w), float(y)),
            "e": (float(x + w), my),
            "se": (float(x + w), float(y + h)),
            "s": (mx, float(y + h)),
            "sw": (float(x), float(y + h)),
            "w": (float(x), my),
        }

    def _hit_test(self, cx: float, cy: float, zoom: float) -> Optional[str]:
        if self.crop_rect is None:
            return None
        hit_r = 10.0 / max(float(zoom), 1e-3)
        for name, (hx, hy) in self._handle_positions().items():
            if math.hypot(cx - hx, cy - hy) <= hit_r:
                return name
        x, y, w, h = self.crop_rect
        if x <= cx <= x + w and y <= cy <= y + h:
            return "move"
        return None

    def _update_drag(
        self,
        doc: "DocumentSession",
        cx: float,
        cy: float,
        modifiers: int,
    ) -> None:
        mods = extract_modifiers(modifiers)
        ratio_pair = None if (mods & MOD_SHIFT) else self.context.crop_aspect_ratio
        target = (ratio_pair[0] / ratio_pair[1]) if (ratio_pair is not None and ratio_pair[1] > 0) else None

        if self._drag_mode == "create":
            self.crop_rect = self._constrain_from_anchor(
                doc, self._start[0], self._start[1], cx, cy, target
            )
            return

        if self._orig_rect is None:
            return
        ox, oy, ow, oh = self._orig_rect

        if self._drag_mode == "move":
            dx = int(round(cx - self._start[0]))
            dy = int(round(cy - self._start[1]))
            nx = max(0, min(doc.width - ow, ox + dx))
            ny = max(0, min(doc.height - oh, oy + dy))
            self.crop_rect = (nx, ny, ow, oh)
            return

        # Corner handles use the opposite corner as anchor
        corner_anchors = {
            "nw": (float(ox + ow), float(oy + oh)),
            "ne": (float(ox), float(oy + oh)),
            "se": (float(ox), float(oy)),
            "sw": (float(ox + ow), float(oy)),
        }
        if self._drag_mode in corner_anchors:
            ax, ay = corner_anchors[self._drag_mode]
            self.crop_rect = self._constrain_from_anchor(doc, ax, ay, cx, cy, target)
            return

        # Edge handles
        x0, y0, x1, y1 = float(ox), float(oy), float(ox + ow), float(oy + oh)
        if self._drag_mode == "w":
            x0 = max(0.0, min(x1 - 1.0, cx))
        elif self._drag_mode == "e":
            x1 = min(float(doc.width), max(x0 + 1.0, cx))
        elif self._drag_mode == "n":
            y0 = max(0.0, min(y1 - 1.0, cy))
        elif self._drag_mode == "s":
            y1 = min(float(doc.height), max(y0 + 1.0, cy))

        nw = x1 - x0
        nh = y1 - y0
        if target is not None and target > 0.0:
            if self._drag_mode in ("w", "e"):
                nh = nw / target
                if nh > float(doc.height):
                    nh = float(doc.height)
                    nw = nh * target
                    if self._drag_mode == "w":
                        x0 = x1 - nw
                    else:
                        x1 = x0 + nw
                cy_mid = oy + oh * 0.5
                y0 = max(0.0, min(float(doc.height) - nh, cy_mid - nh * 0.5))
                y1 = y0 + nh
            else:
                nw = nh * target
                if nw > float(doc.width):
                    nw = float(doc.width)
                    nh = nw / target
                    if self._drag_mode == "n":
                        y0 = y1 - nh
                    else:
                        y1 = y0 + nh
                cx_mid = ox + ow * 0.5
                x0 = max(0.0, min(float(doc.width) - nw, cx_mid - nw * 0.5))
                x1 = x0 + nw

        ix0 = max(0, min(doc.width, int(round(x0))))
        iy0 = max(0, min(doc.height, int(round(y0))))
        ix1 = max(0, min(doc.width, int(round(x1))))
        iy1 = max(0, min(doc.height, int(round(y1))))
        if ix1 > ix0 and iy1 > iy0:
            self.crop_rect = (ix0, iy0, ix1 - ix0, iy1 - iy0)

    def apply_aspect_ratio_to_current_rect(self, doc: Optional["DocumentSession"]) -> None:
        """Fit existing crop_rect (or full canvas if none) to context.crop_aspect_ratio."""
        if doc is None:
            return
        ratio_pair = self.context.crop_aspect_ratio
        if ratio_pair is None or ratio_pair[1] <= 0:
            self.context.notify_canvas_invalidated()
            return
        target = ratio_pair[0] / ratio_pair[1]

        if self.crop_rect is None:
            base_w, base_h = float(doc.width), float(doc.height)
            mid_x, mid_y = base_w * 0.5, base_h * 0.5
        else:
            rx, ry, rw, rh = self.crop_rect
            base_w, base_h = float(rw), float(rh)
            mid_x, mid_y = rx + rw * 0.5, ry + rh * 0.5

        if base_w / max(base_h, 1e-6) >= target:
            nh = base_h
            nw = nh * target
        else:
            nw = base_w
            nh = nw / target

        if nw > doc.width:
            nw = float(doc.width)
            nh = nw / target
        if nh > doc.height:
            nh = float(doc.height)
            nw = nh * target

        cw = max(1, min(doc.width, int(round(nw))))
        ch = max(1, min(doc.height, int(round(nh))))
        x0 = max(0, min(doc.width - cw, int(round(mid_x - cw * 0.5))))
        y0 = max(0, min(doc.height - ch, int(round(mid_y - ch * 0.5))))
        self.crop_rect = (x0, y0, cw, ch)
        self.context.notify_canvas_invalidated()

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
        hit = self._hit_test(float(cx), float(cy), doc.viewport.zoom)
        self._dragging = True
        self._start = (float(cx), float(cy))
        if hit is not None and self.crop_rect is not None:
            self._drag_mode = hit
            self._orig_rect = self.crop_rect
        else:
            self._drag_mode = "create"
            self._orig_rect = None
            self.crop_rect = None
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
        self._update_drag(doc, float(cx), float(cy), modifiers)
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
        self._update_drag(doc, float(cx), float(cy), modifiers)
        self._orig_rect = None
        self.context.notify_canvas_invalidated()

    def apply_crop(self, doc: "DocumentSession") -> bool:
        """Apply the pending crop_rect to the document as an undoable CallableCommand."""
        if self.crop_rect is None:
            return False
        rx, ry, rw, rh = self.crop_rect
        x0 = max(0, min(doc.width, int(rx)))
        y0 = max(0, min(doc.height, int(ry)))
        x1 = max(0, min(doc.width, int(rx + rw)))
        y1 = max(0, min(doc.height, int(ry + rh)))
        cw, ch = x1 - x0, y1 - y0
        if cw <= 0 or ch <= 0:
            self.crop_rect = None
            return False

        old_w, old_h = doc.width, doc.height
        layer_offsets = [(layer, layer.offset_x, layer.offset_y) for layer in doc.layers]
        old_sel = doc.selection_mask

        def do(d: "DocumentSession") -> None:
            d.width = cw
            d.height = ch
            for layer, ox, oy in layer_offsets:
                layer.offset_x = ox - x0
                layer.offset_y = oy - y0
            d.selection_mask = None
            d._composited_cache = None
            d._clip_alpha_scratch = None

        def undo(d: "DocumentSession") -> None:
            d.width = old_w
            d.height = old_h
            for layer, ox, oy in layer_offsets:
                layer.offset_x = ox
                layer.offset_y = oy
            d.selection_mask = old_sel
            d._composited_cache = None
            d._clip_alpha_scratch = None

        do(doc)
        doc.undo_stack.push(
            CallableCommand(
                undo_fn=undo,
                redo_fn=do,
                description=f"Crop ({cw}x{ch})",
            )
        )
        doc.modified = True
        doc.invalidate_composite()
        self.crop_rect = None
        self._dragging = False
        self.context.notify_canvas_invalidated()
        self.context.notify_document_resized()
        return True

    def cancel_crop(self) -> None:
        self.crop_rect = None
        self._dragging = False
        self._orig_rect = None
        self.context.notify_canvas_invalidated()

    def key_press(
        self,
        doc: "DocumentSession",
        key: int,
        modifiers: int = 0,
    ) -> bool:
        k = extract_key(key)
        if k in (_KEY_RETURN, _KEY_ENTER):
            return self.apply_crop(doc)
        if k == _KEY_ESCAPE:
            self.cancel_crop()
            return True
        return False

    def draw_overlay(
        self,
        painter: Any,
        doc: "DocumentSession",
        viewport: "ViewportTransform",
    ) -> None:
        if self.crop_rect is None or painter is None:
            return
        from PySide6.QtCore import QRectF
        from PySide6.QtGui import QColor, QPainterPath, QPen

        cx, cy, cw, ch = self.crop_rect
        sx0, sy0 = viewport.canvas_to_screen(cx, cy)
        sx1, sy1 = viewport.canvas_to_screen(cx + cw, cy + ch)
        dx0, dy0 = viewport.canvas_to_screen(0, 0)
        dx1, dy1 = viewport.canvas_to_screen(doc.width, doc.height)

        crop_qrect = QRectF(sx0, sy0, sx1 - sx0, sy1 - sy0)
        doc_qrect = QRectF(dx0, dy0, dx1 - dx0, dy1 - dy0)

        painter.save()
        shade = QPainterPath()
        shade.addRect(doc_qrect)
        inner = QPainterPath()
        inner.addRect(crop_qrect)
        painter.fillPath(shade.subtracted(inner), QColor(0, 0, 0, 128))

        painter.setPen(QPen(QColor(255, 255, 255, 240), 1.5))
        painter.drawRect(crop_qrect)

        # Rule of thirds guides
        painter.setPen(QPen(QColor(255, 255, 255, 110), 1.0))
        for frac in (1.0 / 3.0, 2.0 / 3.0):
            gx = crop_qrect.left() + crop_qrect.width() * frac
            gy = crop_qrect.top() + crop_qrect.height() * frac
            painter.drawLine(int(gx), int(crop_qrect.top()), int(gx), int(crop_qrect.bottom()))
            painter.drawLine(int(crop_qrect.left()), int(gy), int(crop_qrect.right()), int(gy))

        # Corner & edge handles
        painter.setBrush(QColor(255, 255, 255, 245))
        painter.setPen(QPen(QColor(0, 120, 215, 255), 1.25))
        for hx, hy in self._handle_positions().values():
            hsx, hsy = viewport.canvas_to_screen(hx, hy)
            painter.drawRect(QRectF(hsx - 4.0, hsy - 4.0, 8.0, 8.0))
        painter.restore()

    def on_deactivate(self, doc: Optional["DocumentSession"] = None) -> None:
        self.cancel_crop()


class FreeTransformTool(BaseTool):
    """Interactive bounding-box scale/rotate tool with Shift aspect-lock and Enter/Esc commit."""

    tool_id = ToolType.TRANSFORM.value
    name = "Free Transform"
    cursor_hint = "size_all"

    def __init__(self, context: ToolContext) -> None:
        super().__init__(context)
        self.active = False
        self.center_x = 0.0
        self.center_y = 0.0
        self.box_w = 0.0
        self.box_h = 0.0
        self.angle_deg = 0.0
        self.orig_w = 0.0
        self.orig_h = 0.0

        self._layer_id: Optional[str] = None
        self._src_pixels: Optional[np.ndarray] = None
        self._src_mask: Optional[np.ndarray] = None
        self._drag_mode: Optional[str] = None
        self._drag_start_canvas: tuple[float, float] = (0.0, 0.0)
        self._drag_orig_state: tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)
        self._drag_start_angle_mouse = 0.0
        self._preview_qimage: Any = None

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return (
            self.center_x - self.box_w * 0.5,
            self.center_y - self.box_h * 0.5,
            self.box_w,
            self.box_h,
        )

    def begin_transform(self, doc: "DocumentSession") -> bool:
        """Initialize interactive transform state for doc.active_layer."""
        layer = doc.active_layer
        if layer is None or layer.locked:
            self.cancel()
            return False

        content = layer.get_non_transparent_bounds()
        if content is not None:
            bx, by, bw, bh = content
        else:
            bx, by, bw, bh = layer.bounds

        lx0 = max(0, bx - layer.offset_x)
        ly0 = max(0, by - layer.offset_y)
        lx1 = min(layer.width, lx0 + bw)
        ly1 = min(layer.height, ly0 + bh)
        if lx1 <= lx0 or ly1 <= ly0:
            self.cancel()
            return False

        self._layer_id = layer.id
        self._src_pixels = np.ascontiguousarray(layer.pixels[ly0:ly1, lx0:lx1].copy())
        if layer.mask is not None:
            self._src_mask = np.ascontiguousarray(layer.mask.buffer[ly0:ly1, lx0:lx1].copy())
        else:
            self._src_mask = None

        self.orig_w = float(lx1 - lx0)
        self.orig_h = float(ly1 - ly0)
        self.box_w = self.orig_w
        self.box_h = self.orig_h
        self.center_x = float(layer.offset_x + lx0) + self.box_w * 0.5
        self.center_y = float(layer.offset_y + ly0) + self.box_h * 0.5
        self.angle_deg = 0.0
        self.active = True
        self._drag_mode = None
        self._preview_qimage = None
        self.context.notify_canvas_invalidated()
        return True

    def set_transform(
        self,
        scale_x: Optional[float] = None,
        scale_y: Optional[float] = None,
        angle_deg: Optional[float] = None,
        dx: float = 0.0,
        dy: float = 0.0,
    ) -> None:
        """Programmatically adjust scale, rotation, or translation before commit."""
        if not self.active:
            return
        if scale_x is not None:
            self.box_w = max(1.0, self.orig_w * float(scale_x))
        if scale_y is not None:
            self.box_h = max(1.0, self.orig_h * float(scale_y))
        if angle_deg is not None:
            self.angle_deg = float(angle_deg)
        self.center_x += float(dx)
        self.center_y += float(dy)
        self.context.notify_canvas_invalidated()

    def _local_to_canvas(self, lx: float, ly: float) -> tuple[float, float]:
        rad = math.radians(self.angle_deg)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        return (
            self.center_x + lx * cos_a - ly * sin_a,
            self.center_y + lx * sin_a + ly * cos_a,
        )

    def _canvas_to_local(self, cx: float, cy: float) -> tuple[float, float]:
        rad = math.radians(self.angle_deg)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        dx = cx - self.center_x
        dy = cy - self.center_y
        return (dx * cos_a + dy * sin_a, -dx * sin_a + dy * cos_a)

    def _handle_positions_local(self, zoom: float) -> dict[str, tuple[float, float]]:
        hw = self.box_w * 0.5
        hh = self.box_h * 0.5
        rot_offset = 24.0 / max(float(zoom), 1e-3)
        return {
            "nw": (-hw, -hh),
            "n": (0.0, -hh),
            "ne": (hw, -hh),
            "e": (hw, 0.0),
            "se": (hw, hh),
            "s": (0.0, hh),
            "sw": (-hw, hh),
            "w": (-hw, 0.0),
            "rotate": (0.0, -hh - rot_offset),
        }

    def _hit_test(self, cx: float, cy: float, zoom: float) -> Optional[str]:
        hit_r = 10.0 / max(float(zoom), 1e-3)
        lx, ly = self._canvas_to_local(cx, cy)
        for name, (hx, hy) in self._handle_positions_local(zoom).items():
            if math.hypot(lx - hx, ly - hy) <= hit_r:
                return name
        if abs(lx) <= self.box_w * 0.5 and abs(ly) <= self.box_h * 0.5:
            return "move"
        return None

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
        layer = doc.active_layer
        if layer is None or layer.locked:
            return
        if not self.active or self._layer_id != layer.id:
            if not self.begin_transform(doc):
                return

        mode = self._hit_test(float(cx), float(cy), doc.viewport.zoom)
        if mode is None:
            return

        self._drag_mode = mode
        self._drag_start_canvas = (float(cx), float(cy))
        self._drag_orig_state = (
            self.center_x,
            self.center_y,
            self.box_w,
            self.box_h,
            self.angle_deg,
        )
        self._drag_start_angle_mouse = math.degrees(
            math.atan2(float(cy) - self.center_y, float(cx) - self.center_x)
        )

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
        if not self.active or self._drag_mode is None or (btn != 0 and (btn & 1) == 0):
            return

        ocx, ocy, ow, oh, oang = self._drag_orig_state
        mods = extract_modifiers(modifiers)

        if self._drag_mode == "move":
            self.center_x = ocx + (float(cx) - self._drag_start_canvas[0])
            self.center_y = ocy + (float(cy) - self._drag_start_canvas[1])
            self.context.notify_canvas_invalidated()
            return

        if self._drag_mode == "rotate":
            cur_ang = math.degrees(math.atan2(float(cy) - ocy, float(cx) - ocx))
            new_ang = oang + (cur_ang - self._drag_start_angle_mouse)
            if mods & MOD_SHIFT:
                new_ang = round(new_ang / 15.0) * 15.0
            self.angle_deg = new_ang
            self.context.notify_canvas_invalidated()
            return

        # Scale handles: anchor opposite corner/edge in unrotated box space
        rad = math.radians(oang)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)

        anchor_sign_x = 0.0
        anchor_sign_y = 0.0
        if "e" in self._drag_mode:
            anchor_sign_x = -0.5
        elif "w" in self._drag_mode:
            anchor_sign_x = 0.5
        if "s" in self._drag_mode:
            anchor_sign_y = -0.5
        elif "n" in self._drag_mode:
            anchor_sign_y = 0.5

        alx = anchor_sign_x * ow
        aly = anchor_sign_y * oh
        anchor_wx = ocx + alx * cos_a - aly * sin_a
        anchor_wy = ocy + alx * sin_a + aly * cos_a

        dx_w = float(cx) - anchor_wx
        dy_w = float(cy) - anchor_wy
        rel_lx = dx_w * cos_a + dy_w * sin_a
        rel_ly = -dx_w * sin_a + dy_w * cos_a

        new_w = ow
        new_h = oh
        if "e" in self._drag_mode:
            new_w = max(1.0, rel_lx)
        elif "w" in self._drag_mode:
            new_w = max(1.0, -rel_lx)
        if "s" in self._drag_mode:
            new_h = max(1.0, rel_ly)
        elif "n" in self._drag_mode:
            new_h = max(1.0, -rel_ly)

        if mods & MOD_SHIFT:
            aspect = self.orig_w / max(1.0, self.orig_h)
            if self._drag_mode in ("w", "e"):
                new_h = max(1.0, new_w / aspect)
            elif self._drag_mode in ("n", "s"):
                new_w = max(1.0, new_h * aspect)
            else:
                sx_ratio = new_w / max(1.0, self.orig_w)
                sy_ratio = new_h / max(1.0, self.orig_h)
                scale = sx_ratio if abs(sx_ratio - 1.0) >= abs(sy_ratio - 1.0) else sy_ratio
                new_w = max(1.0, self.orig_w * scale)
                new_h = max(1.0, self.orig_h * scale)

        # New center relative to anchor
        c_off_lx = (-anchor_sign_x) * new_w
        c_off_ly = (-anchor_sign_y) * new_h
        self.box_w = new_w
        self.box_h = new_h
        self.center_x = anchor_wx + c_off_lx * cos_a - c_off_ly * sin_a
        self.center_y = anchor_wy + c_off_lx * sin_a + c_off_ly * cos_a
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
        self._drag_mode = None

    def commit(self, doc: "DocumentSession") -> bool:
        """Commit transformed pixels (and mask if present) via bilinear resampling with Undo."""
        if not self.active or self._layer_id is None or self._src_pixels is None:
            return False
        layer = doc.get_layer_by_id(self._layer_id)
        if layer is None or layer.locked:
            self.cancel()
            return False

        angle_mod = self.angle_deg % 360.0
        if abs(angle_mod) < 1e-4 or abs(angle_mod - 360.0) < 1e-4:
            tw = max(1, int(round(self.box_w)))
            th = max(1, int(round(self.box_h)))
            tx0 = int(round(self.center_x - tw * 0.5))
            ty0 = int(round(self.center_y - th * 0.5))
            transformed_rgba = get_engine().bilinear_resize(self._src_pixels, tw, th)
            transformed_mask: Optional[np.ndarray] = None
            if self._src_mask is not None:
                m = LayerMask(self._src_mask.shape[1], self._src_mask.shape[0], buffer=self._src_mask.copy())
                m.resize(tw, th)
                transformed_mask = m.buffer
        else:
            tx0, ty0, tw, th, transformed_rgba = _bilinear_resample_rotated_rgba(
                self._src_pixels,
                self.box_w,
                self.box_h,
                self.center_x,
                self.center_y,
                self.angle_deg,
            )
            transformed_mask = None
            if self._src_mask is not None:
                transformed_mask = _bilinear_resample_rotated_mask(
                    self._src_mask,
                    self.box_w,
                    self.box_h,
                    self.center_x,
                    self.center_y,
                    self.angle_deg,
                    tx0,
                    ty0,
                    tw,
                    th,
                )

        # Keep at least existing layer bounds so canvas-sized layers do not shrink
        new_ox = min(layer.offset_x, tx0)
        new_oy = min(layer.offset_y, ty0)
        new_x1 = max(layer.offset_x + layer.width, tx0 + tw)
        new_y1 = max(layer.offset_y + layer.height, ty0 + th)
        new_w = new_x1 - new_ox
        new_h = new_y1 - new_oy

        new_pixels = np.zeros((new_h, new_w, 4), dtype=np.uint8)
        px_x = tx0 - new_ox
        px_y = ty0 - new_oy
        new_pixels[px_y : px_y + th, px_x : px_x + tw] = transformed_rgba

        old_pixels = layer.pixels
        old_ox, old_oy = layer.offset_x, layer.offset_y
        old_mask = layer.mask

        if transformed_mask is not None:
            new_mask_buf = np.zeros((new_h, new_w), dtype=np.uint8)
            new_mask_buf[px_y : px_y + th, px_x : px_x + tw] = transformed_mask
            new_mask: Optional[LayerMask] = LayerMask(
                new_w,
                new_h,
                buffer=new_mask_buf,
                enabled=old_mask.enabled if old_mask is not None else True,
            )
        else:
            new_mask = old_mask

        def do(d: "DocumentSession") -> None:
            layer.pixels = new_pixels
            layer.offset_x = new_ox
            layer.offset_y = new_oy
            layer.mask = new_mask

        def undo(d: "DocumentSession") -> None:
            layer.pixels = old_pixels
            layer.offset_x = old_ox
            layer.offset_y = old_oy
            layer.mask = old_mask

        do(doc)
        doc.undo_stack.push(
            CallableCommand(
                undo_fn=undo,
                redo_fn=do,
                description="Free Transform",
                estimated_bytes=int(old_pixels.nbytes + new_pixels.nbytes),
            )
        )
        doc.modified = True
        doc.invalidate_composite()
        self.cancel()
        return True

    def cancel(self) -> None:
        self.active = False
        self._layer_id = None
        self._src_pixels = None
        self._src_mask = None
        self._drag_mode = None
        self._preview_qimage = None
        self.context.notify_canvas_invalidated()

    def key_press(
        self,
        doc: "DocumentSession",
        key: int,
        modifiers: int = 0,
    ) -> bool:
        if not self.active:
            return False
        k = extract_key(key)
        if k in (_KEY_RETURN, _KEY_ENTER):
            return self.commit(doc)
        if k == _KEY_ESCAPE:
            self.cancel()
            return True
        return False

    def draw_overlay(
        self,
        painter: Any,
        doc: "DocumentSession",
        viewport: "ViewportTransform",
    ) -> None:
        if not self.active or painter is None:
            return
        from PySide6.QtCore import QPointF, QRectF
        from PySide6.QtGui import QColor, QImage, QPen, QPolygonF

        painter.save()

        # Draw live transformed preview of source patch
        if self._src_pixels is not None:
            if self._preview_qimage is None:
                sh, sw = self._src_pixels.shape[0], self._src_pixels.shape[1]
                self._preview_qimage = QImage(
                    self._src_pixels.data,
                    sw,
                    sh,
                    sw * 4,
                    QImage.Format.Format_RGBA8888,
                )
            scx, scy = viewport.canvas_to_screen(self.center_x, self.center_y)
            painter.save()
            painter.translate(scx, scy)
            painter.rotate(self.angle_deg)
            sw_screen = self.box_w * viewport.zoom
            sh_screen = self.box_h * viewport.zoom
            painter.setRenderHint(painter.RenderHint.SmoothPixmapTransform, True)
            painter.drawImage(
                QRectF(-sw_screen * 0.5, -sh_screen * 0.5, sw_screen, sh_screen),
                self._preview_qimage,
            )
            painter.restore()

        # Draw bounding box polygon
        hw = self.box_w * 0.5
        hh = self.box_h * 0.5
        corners_screen = [
            QPointF(*viewport.canvas_to_screen(*self._local_to_canvas(lx, ly)))
            for lx, ly in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))
        ]
        painter.setPen(QPen(QColor(0, 170, 255, 240), 1.5))
        painter.drawPolygon(QPolygonF(corners_screen))

        # Rotation stem
        handles_local = self._handle_positions_local(viewport.zoom)
        top_mid = QPointF(*viewport.canvas_to_screen(*self._local_to_canvas(*handles_local["n"])))
        rot_pt = QPointF(*viewport.canvas_to_screen(*self._local_to_canvas(*handles_local["rotate"])))
        painter.drawLine(top_mid, rot_pt)

        # Handles
        painter.setBrush(QColor(255, 255, 255, 255))
        painter.setPen(QPen(QColor(0, 120, 215, 255), 1.5))
        for name, (lx, ly) in handles_local.items():
            sx, sy = viewport.canvas_to_screen(*self._local_to_canvas(lx, ly))
            if name == "rotate":
                painter.drawEllipse(QPointF(sx, sy), 5.0, 5.0)
            else:
                painter.drawRect(QRectF(sx - 4.0, sy - 4.0, 8.0, 8.0))

        painter.restore()

    def on_activate(self, doc: Optional["DocumentSession"] = None) -> None:
        if doc is not None:
            self.begin_transform(doc)

    def on_deactivate(self, doc: Optional["DocumentSession"] = None) -> None:
        self.cancel()
