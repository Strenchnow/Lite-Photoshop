"""Drawing and retouch tools: Brush, Eraser, Smudge, and Local Blur/Sharpen."""

from __future__ import annotations

import math
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
    merge_rect,
)
from app.tools.utility_tools import pick_color_at

if TYPE_CHECKING:
    from app.core.document import DocumentSession, ViewportTransform
    from app.core.layer import Layer


def _mask_paint_value(fg_color: tuple[int, int, int, int]) -> int:
    r, g, b, _ = fg_color
    lum = int(round(0.299 * r + 0.587 * g + 0.114 * b))
    return max(0, min(255, lum))


def _advance_stroke_start(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    radius: float,
    spacing: float,
) -> Optional[tuple[float, float]]:
    dx = x1 - x0
    dy = y1 - y0
    dist = math.hypot(dx, dy)
    if dist < 1e-3:
        return None
    step = max(1.0, float(radius) * min(2.0, max(0.05, float(spacing))))
    n_steps = max(1, int(math.ceil(dist / step)))
    return (x0 + dx / n_steps, y0 + dy / n_steps)


class _BaseStrokeTool(BaseTool):
    """Shared stroke lifecycle, mask-target routing, and minimal DeltaPatchCommand undo."""

    cursor_hint = "cross"
    _supports_mask: bool = False
    _allow_alt_eyedropper: bool = False

    def __init__(self, context: ToolContext) -> None:
        super().__init__(context)
        self._drawing = False
        self._stroke_layer_id: Optional[str] = None
        self._stroke_target: str = "pixels"
        self._before_full: Optional[np.ndarray] = None
        self._dirty_bbox: Optional[tuple[int, int, int, int]] = None
        self._last_local: tuple[float, float] = (0.0, 0.0)
        self._aligned_sel: Optional[np.ndarray] = None
        self._hover_canvas: Optional[tuple[float, float]] = None

    def _begin_stamp(
        self,
        layer: "Layer",
        buf: np.ndarray,
        lx: float,
        ly: float,
    ) -> Optional[tuple[int, int, int, int]]:
        raise NotImplementedError

    def _move_stroke(
        self,
        layer: "Layer",
        buf: np.ndarray,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
    ) -> Optional[tuple[int, int, int, int]]:
        raise NotImplementedError

    def _commit_stroke(self, doc: Optional["DocumentSession"]) -> None:
        if not self._drawing:
            return
        self._drawing = False

        if (
            doc is not None
            and self._stroke_layer_id is not None
            and self._before_full is not None
            and self._dirty_bbox is not None
        ):
            layer = doc.get_layer_by_id(self._stroke_layer_id)
            if layer is not None:
                dx, dy, dw, dh = self._dirty_bbox
                if self._stroke_target == "mask" and layer.mask is not None:
                    buf = layer.mask.buffer
                else:
                    buf = layer.pixels
                if (
                    0 <= dx
                    and 0 <= dy
                    and dx + dw <= buf.shape[1]
                    and dy + dh <= buf.shape[0]
                ):
                    b_roi = self._before_full[dy : dy + dh, dx : dx + dw]
                    a_roi = buf[dy : dy + dh, dx : dx + dw]
                    rel = compute_dirty_rect(b_roi, a_roi)
                    if rel is not None:
                        rx, ry, rw, rh = rel
                        abs_rect = (dx + rx, dy + ry, rw, rh)
                        ax, ay = abs_rect[0], abs_rect[1]
                        doc.record_delta_patch(
                            layer_id=layer.id,
                            dirty_rect=abs_rect,
                            before_patch=self._before_full[ay : ay + rh, ax : ax + rw],
                            after_patch=buf[ay : ay + rh, ax : ax + rw],
                            target=self._stroke_target,
                            description=self.name or "Stroke",
                        )

        self._before_full = None
        self._aligned_sel = None
        self._dirty_bbox = None
        self._stroke_layer_id = None
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
        self._hover_canvas = (float(cx), float(cy))
        if extract_button(button) != 1:
            return
        if self._allow_alt_eyedropper and (extract_modifiers(modifiers) & MOD_ALT):
            pick_color_at(doc, self.context, cx, cy)
            return

        layer = doc.active_layer
        if layer is None or layer.locked:
            return

        if self._supports_mask and self.context.edit_mask and layer.mask is not None:
            target = "mask"
            buf = layer.mask.buffer
        else:
            target = "pixels"
            buf = layer.pixels

        self._drawing = True
        self._stroke_layer_id = layer.id
        self._stroke_target = target
        self._before_full = buf.copy()
        self._dirty_bbox = None
        self._aligned_sel = align_selection_mask_to_layer(doc, layer)

        lx = float(cx) - layer.offset_x
        ly = float(cy) - layer.offset_y
        self._last_local = (lx, ly)

        dirty = self._begin_stamp(layer, buf, lx, ly)
        self._dirty_bbox = merge_rect(self._dirty_bbox, dirty)
        if dirty is not None:
            doc.invalidate_composite()
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
        self._hover_canvas = (float(cx), float(cy))
        btn = extract_button(buttons)
        if not self._drawing or (btn != 0 and (btn & 1) == 0):
            self.context.notify_canvas_invalidated()
            return

        if self._stroke_layer_id is None:
            return
        layer = doc.get_layer_by_id(self._stroke_layer_id)
        if layer is None or layer.locked:
            return

        if self._stroke_target == "mask":
            if layer.mask is None:
                return
            buf = layer.mask.buffer
        else:
            buf = layer.pixels

        lx = float(cx) - layer.offset_x
        ly = float(cy) - layer.offset_y
        dirty = self._move_stroke(layer, buf, self._last_local[0], self._last_local[1], lx, ly)
        self._dirty_bbox = merge_rect(self._dirty_bbox, dirty)
        if math.hypot(lx - self._last_local[0], ly - self._last_local[1]) >= 1e-3:
            self._last_local = (lx, ly)
        if dirty is not None:
            doc.invalidate_composite()
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
        self._hover_canvas = (float(cx), float(cy))
        self._commit_stroke(doc)

    def on_deactivate(self, doc: Optional["DocumentSession"] = None) -> None:
        self._commit_stroke(doc)
        self._hover_canvas = None

    def draw_overlay(
        self,
        painter: Any,
        doc: "DocumentSession",
        viewport: "ViewportTransform",
    ) -> None:
        if painter is None or self._hover_canvas is None:
            return
        from PySide6.QtCore import QPointF
        from PySide6.QtGui import QColor, QPen

        sx, sy = viewport.canvas_to_screen(*self._hover_canvas)
        r = max(1.0, float(self.context.brush_radius) * float(viewport.zoom))
        painter.save()
        painter.setPen(QPen(QColor(0, 0, 0, 180), 2.0))
        painter.drawEllipse(QPointF(sx, sy), r, r)
        painter.setPen(QPen(QColor(255, 255, 255, 220), 1.0))
        painter.drawEllipse(QPointF(sx, sy), r, r)
        painter.restore()


class BrushTool(_BaseStrokeTool):
    """Smooth interpolated brush painting RGBA layers or grayscale LayerMasks."""

    tool_id = ToolType.BRUSH.value
    name = "Brush"
    _supports_mask = True
    _allow_alt_eyedropper = True

    def _begin_stamp(
        self,
        layer: "Layer",
        buf: np.ndarray,
        lx: float,
        ly: float,
    ) -> Optional[tuple[int, int, int, int]]:
        engine = get_engine()
        if self._stroke_target == "mask":
            val = _mask_paint_value(self.context.fg_color)
            return engine.brush_stroke_mask(
                buf,
                x0=lx,
                y0=ly,
                x1=lx,
                y1=ly,
                radius=float(self.context.brush_radius),
                hardness=float(self.context.brush_hardness),
                spacing=float(self.context.brush_spacing),
                value=val,
                opacity=float(self.context.brush_opacity),
                flow=float(self.context.brush_flow),
                selection_mask=self._aligned_sel,
            )
        return engine.brush_dab(
            buf,
            cx=lx,
            cy=ly,
            radius=float(self.context.brush_radius),
            hardness=float(self.context.brush_hardness),
            color=self.context.fg_color,
            opacity=float(self.context.brush_opacity),
            flow=float(self.context.brush_flow),
            is_eraser=False,
            selection_mask=self._aligned_sel,
        )

    def _move_stroke(
        self,
        layer: "Layer",
        buf: np.ndarray,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
    ) -> Optional[tuple[int, int, int, int]]:
        start = _advance_stroke_start(
            x0, y0, x1, y1, self.context.brush_radius, self.context.brush_spacing
        )
        if start is None:
            return None
        sx, sy = start
        engine = get_engine()
        if self._stroke_target == "mask":
            val = _mask_paint_value(self.context.fg_color)
            return engine.brush_stroke_mask(
                buf,
                x0=sx,
                y0=sy,
                x1=x1,
                y1=y1,
                radius=float(self.context.brush_radius),
                hardness=float(self.context.brush_hardness),
                spacing=float(self.context.brush_spacing),
                value=val,
                opacity=float(self.context.brush_opacity),
                flow=float(self.context.brush_flow),
                selection_mask=self._aligned_sel,
            )
        return engine.brush_stroke(
            buf,
            x0=sx,
            y0=sy,
            x1=x1,
            y1=y1,
            radius=float(self.context.brush_radius),
            hardness=float(self.context.brush_hardness),
            spacing=float(self.context.brush_spacing),
            color=self.context.fg_color,
            opacity=float(self.context.brush_opacity),
            flow=float(self.context.brush_flow),
            is_eraser=False,
            selection_mask=self._aligned_sel,
        )


class EraserTool(_BaseStrokeTool):
    """Erases layer alpha or paints 0 onto an active LayerMask."""

    tool_id = ToolType.ERASER.value
    name = "Eraser"
    _supports_mask = True

    def _begin_stamp(
        self,
        layer: "Layer",
        buf: np.ndarray,
        lx: float,
        ly: float,
    ) -> Optional[tuple[int, int, int, int]]:
        engine = get_engine()
        if self._stroke_target == "mask":
            return engine.brush_stroke_mask(
                buf,
                x0=lx,
                y0=ly,
                x1=lx,
                y1=ly,
                radius=float(self.context.brush_radius),
                hardness=float(self.context.brush_hardness),
                spacing=float(self.context.brush_spacing),
                value=0,
                opacity=float(self.context.brush_opacity),
                flow=float(self.context.brush_flow),
                selection_mask=self._aligned_sel,
            )
        return engine.brush_dab(
            buf,
            cx=lx,
            cy=ly,
            radius=float(self.context.brush_radius),
            hardness=float(self.context.brush_hardness),
            color=(0, 0, 0, 255),
            opacity=float(self.context.brush_opacity),
            flow=float(self.context.brush_flow),
            is_eraser=True,
            selection_mask=self._aligned_sel,
        )

    def _move_stroke(
        self,
        layer: "Layer",
        buf: np.ndarray,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
    ) -> Optional[tuple[int, int, int, int]]:
        start = _advance_stroke_start(
            x0, y0, x1, y1, self.context.brush_radius, self.context.brush_spacing
        )
        if start is None:
            return None
        sx, sy = start
        engine = get_engine()
        if self._stroke_target == "mask":
            return engine.brush_stroke_mask(
                buf,
                x0=sx,
                y0=sy,
                x1=x1,
                y1=y1,
                radius=float(self.context.brush_radius),
                hardness=float(self.context.brush_hardness),
                spacing=float(self.context.brush_spacing),
                value=0,
                opacity=float(self.context.brush_opacity),
                flow=float(self.context.brush_flow),
                selection_mask=self._aligned_sel,
            )
        return engine.brush_stroke(
            buf,
            x0=sx,
            y0=sy,
            x1=x1,
            y1=y1,
            radius=float(self.context.brush_radius),
            hardness=float(self.context.brush_hardness),
            spacing=float(self.context.brush_spacing),
            color=(0, 0, 0, 255),
            opacity=float(self.context.brush_opacity),
            flow=float(self.context.brush_flow),
            is_eraser=True,
            selection_mask=self._aligned_sel,
        )


class SmudgeTool(_BaseStrokeTool):
    """C-accelerated smudge brush carrying and blending pixels along drag vectors."""

    tool_id = ToolType.SMUDGE.value
    name = "Smudge"
    _supports_mask = False

    def _begin_stamp(
        self,
        layer: "Layer",
        buf: np.ndarray,
        lx: float,
        ly: float,
    ) -> Optional[tuple[int, int, int, int]]:
        return None

    def _move_stroke(
        self,
        layer: "Layer",
        buf: np.ndarray,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
    ) -> Optional[tuple[int, int, int, int]]:
        if math.hypot(x1 - x0, y1 - y0) < 1e-3:
            return None
        return get_engine().smudge_stroke(
            buf,
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            radius=float(self.context.brush_radius),
            hardness=float(self.context.brush_hardness),
            strength=float(self.context.smudge_strength),
            selection_mask=self._aligned_sel,
        )


class BlurSharpenTool(_BaseStrokeTool):
    """Localized C Gaussian blur / unsharp-mask sharpen brush."""

    tool_id = ToolType.BLUR_SHARPEN.value
    name = "Blur / Sharpen"
    _supports_mask = False

    def _kernel_size(self) -> int:
        return 5 if int(self.context.filter_kernel_size) >= 5 else 3

    def _begin_stamp(
        self,
        layer: "Layer",
        buf: np.ndarray,
        lx: float,
        ly: float,
    ) -> Optional[tuple[int, int, int, int]]:
        return get_engine().blur_sharpen_dab(
            buf,
            cx=lx,
            cy=ly,
            radius=float(self.context.brush_radius),
            hardness=float(self.context.brush_hardness),
            strength=float(self.context.filter_strength),
            mode=int(self.context.filter_mode),
            kernel_size=self._kernel_size(),
            selection_mask=self._aligned_sel,
        )

    def _move_stroke(
        self,
        layer: "Layer",
        buf: np.ndarray,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
    ) -> Optional[tuple[int, int, int, int]]:
        dx = x1 - x0
        dy = y1 - y0
        dist = math.hypot(dx, dy)
        if dist < 1e-3:
            return None

        radius = float(self.context.brush_radius)
        step = max(1.0, radius * min(2.0, max(0.05, float(self.context.brush_spacing))))
        n_steps = max(1, int(math.ceil(dist / step)))
        engine = get_engine()
        mode = int(self.context.filter_mode)
        ksize = self._kernel_size()
        hardness = float(self.context.brush_hardness)
        strength = float(self.context.filter_strength)

        dirty_union: Optional[tuple[int, int, int, int]] = None
        for i in range(1, n_steps + 1):
            t = float(i) / float(n_steps)
            dirty = engine.blur_sharpen_dab(
                buf,
                cx=x0 + dx * t,
                cy=y0 + dy * t,
                radius=radius,
                hardness=hardness,
                strength=strength,
                mode=mode,
                kernel_size=ksize,
                selection_mask=self._aligned_sel,
            )
            dirty_union = merge_rect(dirty_union, dirty)
        return dirty_union
