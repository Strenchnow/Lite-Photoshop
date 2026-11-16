"""Navigation and positioning tools: HandTool and MoveTool with magnetic snapping."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from app.tools.base_tool import (
    MOD_SHIFT,
    BaseTool,
    ToolContext,
    ToolType,
    extract_button,
    extract_modifiers,
)

if TYPE_CHECKING:
    from app.core.document import DocumentSession


def center_horizontal(doc: "DocumentSession", use_content_bounds: bool = True) -> None:
    """Center the active layer horizontally on the canvas with Undo support."""
    layer = doc.active_layer
    if layer is None or layer.locked:
        return
    doc.center_active_layer(
        horizontal=True,
        vertical=False,
        use_content_bounds=use_content_bounds,
    )


def center_vertical(doc: "DocumentSession", use_content_bounds: bool = True) -> None:
    """Center the active layer vertically on the canvas with Undo support."""
    layer = doc.active_layer
    if layer is None or layer.locked:
        return
    doc.center_active_layer(
        horizontal=False,
        vertical=True,
        use_content_bounds=use_content_bounds,
    )


def center_both(doc: "DocumentSession", use_content_bounds: bool = True) -> None:
    """Center the active layer both horizontally and vertically with Undo support."""
    layer = doc.active_layer
    if layer is None or layer.locked:
        return
    doc.center_active_layer(
        horizontal=True,
        vertical=True,
        use_content_bounds=use_content_bounds,
    )


class HandTool(BaseTool):
    """Pans the document viewport in screen coordinates."""

    tool_id = ToolType.HAND.value
    name = "Hand"
    cursor_hint = "open_hand"

    def __init__(self, context: ToolContext) -> None:
        super().__init__(context)
        self._dragging = False
        self._last_sx = 0.0
        self._last_sy = 0.0

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
        self._last_sx = float(sx)
        self._last_sy = float(sy)
        self.cursor_hint = "closed_hand"

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
        dx = float(sx) - self._last_sx
        dy = float(sy) - self._last_sy
        self._last_sx = float(sx)
        self._last_sy = float(sy)
        if dx != 0.0 or dy != 0.0:
            doc.viewport.pan(dx, dy)
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
        self._dragging = False
        self.cursor_hint = "open_hand"

    def on_deactivate(self, doc: Optional["DocumentSession"] = None) -> None:
        self._dragging = False
        self.cursor_hint = "open_hand"


class MoveTool(BaseTool):
    """Moves the active layer with magnetic snapping to canvas edges and center axes."""

    tool_id = ToolType.MOVE.value
    name = "Move"
    cursor_hint = "size_all"

    def __init__(self, context: ToolContext) -> None:
        super().__init__(context)
        self._dragging = False
        self._layer_id: Optional[str] = None
        self._start_cx = 0.0
        self._start_cy = 0.0
        self._orig_offset_x = 0
        self._orig_offset_y = 0
        self._content_rel_x = 0
        self._content_rel_y = 0
        self._content_w = 0
        self._content_h = 0

    def center_horizontal(self, doc: "DocumentSession") -> None:
        center_horizontal(doc)
        self.context.notify_canvas_invalidated()

    def center_vertical(self, doc: "DocumentSession") -> None:
        center_vertical(doc)
        self.context.notify_canvas_invalidated()

    def center_both(self, doc: "DocumentSession") -> None:
        center_both(doc)
        self.context.notify_canvas_invalidated()

    def _compute_snapped_offset(
        self,
        doc: "DocumentSession",
        raw_ox: int,
        raw_oy: int,
        modifiers: int,
    ) -> tuple[int, int]:
        if extract_modifiers(modifiers) & MOD_SHIFT:
            self.context.snap_guides.clear()
            return raw_ox, raw_oy

        threshold = 6.0 / max(float(doc.viewport.zoom), 1e-3)
        guides: list[tuple[str, float]] = []

        # Horizontal snapping -> vertical guide lines
        left = float(raw_ox + self._content_rel_x)
        hcenter = left + self._content_w / 2.0
        right = left + float(self._content_w)
        x_targets = (0.0, doc.width / 2.0, float(doc.width))

        best_dx: Optional[float] = None
        best_x_target: Optional[float] = None
        best_x_dist = threshold + 1e-9
        for anchor in (left, hcenter, right):
            for target in x_targets:
                dist = abs(target - anchor)
                if dist <= best_x_dist:
                    best_x_dist = dist
                    best_dx = target - anchor
                    best_x_target = target

        snapped_ox = int(round(raw_ox + best_dx)) if best_dx is not None else raw_ox
        if best_x_target is not None:
            s_left = float(snapped_ox + self._content_rel_x)
            s_hcenter = s_left + self._content_w / 2.0
            s_right = s_left + float(self._content_w)
            added_x: set[float] = set()
            for target in x_targets:
                if (
                    target == best_x_target
                    or min(abs(s_left - target), abs(s_hcenter - target), abs(s_right - target)) <= 0.6
                ):
                    if target not in added_x:
                        guides.append(("vertical", float(target)))
                        added_x.add(target)

        # Vertical snapping -> horizontal guide lines
        top = float(raw_oy + self._content_rel_y)
        vcenter = top + self._content_h / 2.0
        bottom = top + float(self._content_h)
        y_targets = (0.0, doc.height / 2.0, float(doc.height))

        best_dy: Optional[float] = None
        best_y_target: Optional[float] = None
        best_y_dist = threshold + 1e-9
        for anchor in (top, vcenter, bottom):
            for target in y_targets:
                dist = abs(target - anchor)
                if dist <= best_y_dist:
                    best_y_dist = dist
                    best_dy = target - anchor
                    best_y_target = target

        snapped_oy = int(round(raw_oy + best_dy)) if best_dy is not None else raw_oy
        if best_y_target is not None:
            s_top = float(snapped_oy + self._content_rel_y)
            s_vcenter = s_top + self._content_h / 2.0
            s_bottom = s_top + float(self._content_h)
            added_y: set[float] = set()
            for target in y_targets:
                if (
                    target == best_y_target
                    or min(abs(s_top - target), abs(s_vcenter - target), abs(s_bottom - target)) <= 0.6
                ):
                    if target not in added_y:
                        guides.append(("horizontal", float(target)))
                        added_y.add(target)

        self.context.snap_guides[:] = guides
        return snapped_ox, snapped_oy

    def _apply_drag(
        self,
        doc: "DocumentSession",
        cx: float,
        cy: float,
        modifiers: int,
    ) -> Optional[tuple[int, int]]:
        if self._layer_id is None:
            return None
        layer = doc.get_layer_by_id(self._layer_id)
        if layer is None or layer.locked:
            return None

        raw_ox = int(round(self._orig_offset_x + (float(cx) - self._start_cx)))
        raw_oy = int(round(self._orig_offset_y + (float(cy) - self._start_cy)))
        new_ox, new_oy = self._compute_snapped_offset(doc, raw_ox, raw_oy, modifiers)
        if layer.offset_x != new_ox or layer.offset_y != new_oy:
            layer.offset_x = new_ox
            layer.offset_y = new_oy
            doc.invalidate_composite()
        self.context.notify_canvas_invalidated()
        return (new_ox, new_oy)

    def _finish_drag(self, doc: Optional["DocumentSession"]) -> None:
        if not self._dragging:
            return
        self._dragging = False
        self.context.snap_guides.clear()

        if doc is not None and self._layer_id is not None:
            idx = doc.get_layer_index(self._layer_id)
            if idx >= 0:
                layer = doc.layers[idx]
                final_x, final_y = layer.offset_x, layer.offset_y
                if final_x != self._orig_offset_x or final_y != self._orig_offset_y:
                    layer.offset_x = self._orig_offset_x
                    layer.offset_y = self._orig_offset_y
                    doc.move_layer_offset(idx, final_x, final_y, description="Move Layer")

        self._layer_id = None
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
        layer = doc.active_layer
        if layer is None or layer.locked:
            return

        content = layer.get_non_transparent_bounds()
        if content is not None:
            cbx, cby, cbw, cbh = content
            self._content_rel_x = cbx - layer.offset_x
            self._content_rel_y = cby - layer.offset_y
            self._content_w = cbw
            self._content_h = cbh
        else:
            self._content_rel_x = 0
            self._content_rel_y = 0
            self._content_w = layer.width
            self._content_h = layer.height

        self._dragging = True
        self._layer_id = layer.id
        self._start_cx = float(cx)
        self._start_cy = float(cy)
        self._orig_offset_x = int(layer.offset_x)
        self._orig_offset_y = int(layer.offset_y)
        self.context.snap_guides.clear()

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
        self._apply_drag(doc, cx, cy, modifiers)

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
        self._apply_drag(doc, cx, cy, modifiers)
        self._finish_drag(doc)

    def on_deactivate(self, doc: Optional["DocumentSession"] = None) -> None:
        self._finish_drag(doc)
