"""DocumentSession, Delta Undo/Redo stack, ViewportTransform, and multi-tab manager."""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional, Protocol

import numpy as np

from app.core.c_bridge import get_engine
from app.core.layer import BlendMode, Layer


def compute_dirty_rect(
    before: np.ndarray,
    after: np.ndarray,
) -> Optional[tuple[int, int, int, int]]:
    """Compute tight (x, y, w, h) bounding box where `before` and `after` differ."""
    if before.shape != after.shape:
        raise ValueError(f"Buffer shape mismatch: {before.shape} != {after.shape}")
    if before.size == 0:
        return None

    if before.ndim == 3 and before.shape[2] == 4:
        if before.flags["C_CONTIGUOUS"] and after.flags["C_CONTIGUOUS"]:
            h, w = before.shape[0], before.shape[1]
            diff_2d = before.view(np.uint32).reshape(h, w) != after.view(np.uint32).reshape(h, w)
        else:
            diff_2d = np.any(before != after, axis=2)
    elif before.ndim == 2:
        diff_2d = before != after
    else:
        raise ValueError(f"Unsupported buffer dims for dirty rect: {before.shape}")

    ys = np.flatnonzero(np.any(diff_2d, axis=1))
    if ys.size == 0:
        return None
    xs = np.flatnonzero(np.any(diff_2d, axis=0))
    x0, x1 = int(xs[0]), int(xs[-1])
    y0, y1 = int(ys[0]), int(ys[-1])
    return (x0, y0, x1 - x0 + 1, y1 - y0 + 1)


class UndoCommand(Protocol):
    """Protocol implemented by undoable document commands."""

    description: str

    @property
    def memory_bytes(self) -> int: ...

    def undo(self, doc: "DocumentSession") -> None: ...

    def redo(self, doc: "DocumentSession") -> None: ...


@dataclass
class DeltaPatchCommand:
    """Memory-efficient undo command storing only the dirty bounding-box patch."""

    layer_id: str
    target: str  # "pixels" or "mask"
    dirty_rect: tuple[int, int, int, int]
    before_patch: np.ndarray
    after_patch: np.ndarray
    description: str = "Edit"

    @property
    def memory_bytes(self) -> int:
        return int(self.before_patch.nbytes + self.after_patch.nbytes)

    def _apply(self, doc: "DocumentSession", patch: np.ndarray) -> None:
        layer = doc.get_layer_by_id(self.layer_id)
        if layer is None:
            raise KeyError(f"Layer '{self.layer_id}' no longer exists in document")
        if self.target == "pixels":
            layer.set_patch(self.dirty_rect, patch)
        elif self.target == "mask":
            if layer.mask is None:
                raise ValueError(f"Layer '{self.layer_id}' has no mask for mask delta command")
            layer.mask.set_patch(self.dirty_rect, patch)
        else:
            raise ValueError(f"Unknown delta target: {self.target!r}")
        doc.invalidate_composite()

    def undo(self, doc: "DocumentSession") -> None:
        self._apply(doc, self.before_patch)

    def redo(self, doc: "DocumentSession") -> None:
        self._apply(doc, self.after_patch)


@dataclass
class CallableCommand:
    """Undoable command for structural or property changes."""

    undo_fn: Callable[["DocumentSession"], None]
    redo_fn: Callable[["DocumentSession"], None]
    description: str = "Action"
    estimated_bytes: int = 0

    @property
    def memory_bytes(self) -> int:
        return int(self.estimated_bytes)

    def undo(self, doc: "DocumentSession") -> None:
        self.undo_fn(doc)
        doc.invalidate_composite()

    def redo(self, doc: "DocumentSession") -> None:
        self.redo_fn(doc)
        doc.invalidate_composite()


class UndoStack:
    """Bounded undo/redo history stack."""

    def __init__(self, max_steps: int = 50) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        self.max_steps = int(max_steps)
        self._undo_list: list[UndoCommand] = []
        self._redo_list: list[UndoCommand] = []

    @property
    def can_undo(self) -> bool:
        return len(self._undo_list) > 0

    @property
    def can_redo(self) -> bool:
        return len(self._redo_list) > 0

    @property
    def undo_count(self) -> int:
        return len(self._undo_list)

    @property
    def redo_count(self) -> int:
        return len(self._redo_list)

    @property
    def total_memory_bytes(self) -> int:
        return sum(c.memory_bytes for c in self._undo_list) + sum(c.memory_bytes for c in self._redo_list)

    def push(self, cmd: UndoCommand) -> None:
        self._undo_list.append(cmd)
        self._redo_list.clear()
        while len(self._undo_list) > self.max_steps:
            self._undo_list.pop(0)

    def undo(self, doc: "DocumentSession") -> Optional[str]:
        if not self._undo_list:
            return None
        cmd = self._undo_list.pop()
        cmd.undo(doc)
        self._redo_list.append(cmd)
        return cmd.description

    def redo(self, doc: "DocumentSession") -> Optional[str]:
        if not self._redo_list:
            return None
        cmd = self._redo_list.pop()
        cmd.redo(doc)
        self._undo_list.append(cmd)
        return cmd.description

    def clear(self) -> None:
        self._undo_list.clear()
        self._redo_list.clear()


@dataclass
class ViewportTransform:
    """2D affine view transform mapping canvas space <-> screen space."""

    zoom: float = 1.0
    pan_x: float = 0.0
    pan_y: float = 0.0
    min_zoom: float = 0.05
    max_zoom: float = 64.0

    def screen_to_canvas(self, screen_x: float, screen_y: float) -> tuple[float, float]:
        return ((screen_x - self.pan_x) / self.zoom, (screen_y - self.pan_y) / self.zoom)

    def canvas_to_screen(self, canvas_x: float, canvas_y: float) -> tuple[float, float]:
        return (canvas_x * self.zoom + self.pan_x, canvas_y * self.zoom + self.pan_y)

    def zoom_at_cursor(self, screen_x: float, screen_y: float, zoom_factor: float) -> float:
        """Scale zoom by `zoom_factor` while keeping the canvas point under cursor stationary."""
        cx, cy = self.screen_to_canvas(screen_x, screen_y)
        new_zoom = max(self.min_zoom, min(self.max_zoom, self.zoom * float(zoom_factor)))
        self.zoom = new_zoom
        self.pan_x = screen_x - cx * new_zoom
        self.pan_y = screen_y - cy * new_zoom
        return new_zoom

    def pan(self, dx: float, dy: float) -> None:
        self.pan_x += float(dx)
        self.pan_y += float(dy)

    def reset_view(
        self,
        viewport_w: float,
        viewport_h: float,
        canvas_w: int,
        canvas_h: int,
        fill_ratio: float = 0.9,
    ) -> None:
        if viewport_w <= 0 or viewport_h <= 0 or canvas_w <= 0 or canvas_h <= 0:
            self.zoom = 1.0
            self.pan_x = 0.0
            self.pan_y = 0.0
            return
        fit_zoom = min(
            (viewport_w * fill_ratio) / canvas_w,
            (viewport_h * fill_ratio) / canvas_h,
        )
        self.zoom = max(self.min_zoom, min(self.max_zoom, fit_zoom))
        self.pan_x = (viewport_w - canvas_w * self.zoom) * 0.5
        self.pan_y = (viewport_h - canvas_h * self.zoom) * 0.5


class DocumentSession:
    """In-memory document model owning layers, selection, viewport, and delta undo/redo."""

    def __init__(
        self,
        width: int,
        height: int,
        title: str = "Untitled",
        layers: Optional[list[Layer]] = None,
        initial_fill: tuple[int, int, int, int] = (0, 0, 0, 0),
        file_path: Optional[Path] = None,
        max_undo_steps: int = 50,
        is_active_tab: bool = True,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError(f"Document dimensions must be positive, got ({width}, {height})")
        self.id = uuid.uuid4().hex
        self.title = str(title)
        self.file_path = Path(file_path) if file_path is not None else None
        self.width = int(width)
        self.height = int(height)

        if layers is not None:
            if not layers:
                raise ValueError("DocumentSession requires at least one layer when layers list is provided")
            self.layers: list[Layer] = list(layers)
        else:
            self.layers = [
                Layer(self.width, self.height, name="Background", fill_rgba=initial_fill)
            ]

        self.active_layer_index: int = len(self.layers) - 1
        self.selection_mask: Optional[np.ndarray] = None
        self.viewport = ViewportTransform()
        self.undo_stack = UndoStack(max_steps=max_undo_steps)
        self.modified: bool = False
        self.is_active_tab: bool = bool(is_active_tab)

        self._composited_cache: Optional[np.ndarray] = None
        self._clip_alpha_scratch: Optional[np.ndarray] = None
        self._composite_dirty: bool = True
        self._pending_edit: Optional[tuple[str, str, tuple[int, int, int, int], np.ndarray]] = None

    @property
    def display_title(self) -> str:
        return f"{self.title} *" if self.modified else self.title

    @property
    def active_layer(self) -> Optional[Layer]:
        if 0 <= self.active_layer_index < len(self.layers):
            return self.layers[self.active_layer_index]
        return None

    def get_layer_by_id(self, layer_id: str) -> Optional[Layer]:
        for layer in self.layers:
            if layer.id == layer_id:
                return layer
        return None

    def get_layer_index(self, layer_id: str) -> int:
        for i, layer in enumerate(self.layers):
            if layer.id == layer_id:
                return i
        return -1

    def set_active_layer(self, index: int) -> Layer:
        if not (0 <= index < len(self.layers)):
            raise IndexError(f"Layer index out of range: {index}")
        self.active_layer_index = int(index)
        return self.layers[self.active_layer_index]

    def invalidate_composite(self) -> None:
        self._composite_dirty = True

    def set_tab_active(self, active: bool) -> None:
        """Release cached composited buffers when tab becomes inactive (zero display overhead)."""
        self.is_active_tab = bool(active)
        if not self.is_active_tab:
            self._composited_cache = None
            self._clip_alpha_scratch = None
            self._composite_dirty = True

    def clear_selection(self) -> None:
        self.selection_mask = None

    def set_selection_mask(self, mask: Optional[np.ndarray]) -> None:
        if mask is None:
            self.selection_mask = None
            return
        arr = np.ascontiguousarray(mask, dtype=np.uint8)
        if arr.shape != (self.height, self.width):
            raise ValueError(
                f"Selection mask shape mismatch: expected ({self.height}, {self.width}), got {arr.shape}"
            )
        self.selection_mask = arr

    def _compute_clip_alpha(self, base_layer: Layer) -> np.ndarray:
        if self._clip_alpha_scratch is None or self._clip_alpha_scratch.shape != (self.height, self.width):
            self._clip_alpha_scratch = np.zeros((self.height, self.width), dtype=np.uint8)
        else:
            self._clip_alpha_scratch.fill(0)

        if not base_layer.visible or base_layer.opacity <= 0.0:
            return self._clip_alpha_scratch

        engine = get_engine()
        has_mask = base_layer.mask is not None and base_layer.mask.enabled
        if (
            base_layer.offset_x == 0
            and base_layer.offset_y == 0
            and base_layer.width == self.width
            and base_layer.height == self.height
            and not has_mask
        ):
            engine.extract_alpha(base_layer.pixels, self._clip_alpha_scratch)
            return self._clip_alpha_scratch

        x0 = max(0, base_layer.offset_x)
        y0 = max(0, base_layer.offset_y)
        x1 = min(self.width, base_layer.offset_x + base_layer.width)
        y1 = min(self.height, base_layer.offset_y + base_layer.height)
        if x1 <= x0 or y1 <= y0:
            return self._clip_alpha_scratch

        sx0 = x0 - base_layer.offset_x
        sy0 = y0 - base_layer.offset_y
        sx1 = sx0 + (x1 - x0)
        sy1 = sy0 + (y1 - y0)

        alpha_slice = base_layer.pixels[sy0:sy1, sx0:sx1, 3]
        if has_mask and base_layer.mask is not None:
            mask_slice = base_layer.mask.buffer[sy0:sy1, sx0:sx1]
            combined = (alpha_slice.astype(np.uint16) * mask_slice.astype(np.uint16) + 127) // 255
            self._clip_alpha_scratch[y0:y1, x0:x1] = combined.astype(np.uint8)
        else:
            self._clip_alpha_scratch[y0:y1, x0:x1] = alpha_slice
        return self._clip_alpha_scratch

    def composite(self, roi: Optional[tuple[int, int, int, int]] = None) -> np.ndarray:
        """Composite all visible layers bottom-to-top using the C pixel engine."""
        if self.is_active_tab and roi is None and not self._composite_dirty and self._composited_cache is not None:
            return self._composited_cache

        engine = get_engine()
        if self.is_active_tab:
            if self._composited_cache is None or self._composited_cache.shape != (self.height, self.width, 4):
                self._composited_cache = np.zeros((self.height, self.width, 4), dtype=np.uint8)
            out = self._composited_cache
        else:
            out = np.zeros((self.height, self.width, 4), dtype=np.uint8)

        if roi is None:
            engine.clear_buffer(out, 0, 0, 0, 0)
        else:
            rx, ry, rw, rh = roi
            x0 = max(0, rx)
            y0 = max(0, ry)
            x1 = min(self.width, rx + rw)
            y1 = min(self.height, ry + rh)
            if x1 > x0 and y1 > y0:
                out[y0:y1, x0:x1] = 0

        current_clip_alpha: Optional[np.ndarray] = None
        current_base_index: int = -1

        for idx, layer in enumerate(self.layers):
            if not layer.clipping_mask or idx == 0:
                current_base_index = idx
                current_clip_alpha = None
                if not layer.visible or layer.opacity <= 0.0:
                    continue
                mask_buf = layer.mask.buffer if (layer.mask is not None and layer.mask.enabled) else None
                engine.composite_layer(
                    dst=out,
                    src=layer.pixels,
                    mask=mask_buf,
                    clip_alpha=None,
                    offset_x=layer.offset_x,
                    offset_y=layer.offset_y,
                    opacity=layer.opacity,
                    blend_mode=int(layer.blend_mode),
                    roi=roi,
                )
            else:
                if not layer.visible or layer.opacity <= 0.0:
                    continue
                if current_clip_alpha is None and current_base_index >= 0:
                    current_clip_alpha = self._compute_clip_alpha(self.layers[current_base_index])
                mask_buf = layer.mask.buffer if (layer.mask is not None and layer.mask.enabled) else None
                engine.composite_layer(
                    dst=out,
                    src=layer.pixels,
                    mask=mask_buf,
                    clip_alpha=current_clip_alpha,
                    offset_x=layer.offset_x,
                    offset_y=layer.offset_y,
                    opacity=layer.opacity,
                    blend_mode=int(layer.blend_mode),
                    roi=roi,
                )

        if self.is_active_tab and roi is None:
            self._composite_dirty = False
        return out

    # --- Delta Undo/Redo Edit Transactions ---

    def begin_layer_edit(
        self,
        layer: Optional[Layer] = None,
        target: str = "pixels",
        roi: Optional[tuple[int, int, int, int]] = None,
    ) -> None:
        """Snapshot a region of a layer or mask before applying pixel mutations."""
        if self._pending_edit is not None:
            raise RuntimeError("A layer edit transaction is already in progress")
        target_layer = layer or self.active_layer
        if target_layer is None:
            raise ValueError("No active layer available for edit")

        if roi is None:
            rect = (0, 0, target_layer.width, target_layer.height)
        else:
            rx, ry, rw, rh = roi
            x0 = max(0, int(rx))
            y0 = max(0, int(ry))
            x1 = min(target_layer.width, int(rx) + int(rw))
            y1 = min(target_layer.height, int(ry) + int(rh))
            rect = (x0, y0, max(0, x1 - x0), max(0, y1 - y0))

        if target == "pixels":
            snapshot = target_layer.get_patch(rect)
        elif target == "mask":
            if target_layer.mask is None:
                raise ValueError(f"Layer '{target_layer.name}' has no mask")
            snapshot = target_layer.mask.get_patch(rect)
        else:
            raise ValueError(f"Invalid edit target: {target!r}")

        self._pending_edit = (target_layer.id, target, rect, snapshot)

    def commit_layer_edit(self, description: str = "Edit") -> Optional[DeltaPatchCommand]:
        """Compute minimal dirty_rect against snapshot and push a DeltaPatchCommand."""
        if self._pending_edit is None:
            raise RuntimeError("No active layer edit transaction to commit")

        layer_id, target, roi_rect, before_snapshot = self._pending_edit
        self._pending_edit = None

        layer = self.get_layer_by_id(layer_id)
        if layer is None or roi_rect[2] <= 0 or roi_rect[3] <= 0:
            return None

        if target == "pixels":
            after_snapshot = layer.get_patch(roi_rect)
        else:
            if layer.mask is None:
                return None
            after_snapshot = layer.mask.get_patch(roi_rect)

        rel_rect = compute_dirty_rect(before_snapshot, after_snapshot)
        if rel_rect is None:
            return None

        rx, ry, rw, rh = rel_rect
        abs_dirty_rect = (roi_rect[0] + rx, roi_rect[1] + ry, rw, rh)
        before_patch = np.ascontiguousarray(before_snapshot[ry : ry + rh, rx : rx + rw].copy())
        after_patch = np.ascontiguousarray(after_snapshot[ry : ry + rh, rx : rx + rw].copy())

        cmd = DeltaPatchCommand(
            layer_id=layer_id,
            target=target,
            dirty_rect=abs_dirty_rect,
            before_patch=before_patch,
            after_patch=after_patch,
            description=description,
        )
        self.undo_stack.push(cmd)
        self.modified = True
        self.invalidate_composite()
        return cmd

    def cancel_layer_edit(self) -> None:
        self._pending_edit = None

    def record_delta_patch(
        self,
        layer_id: str,
        dirty_rect: tuple[int, int, int, int],
        before_patch: np.ndarray,
        after_patch: np.ndarray,
        target: str = "pixels",
        description: str = "Edit",
    ) -> Optional[DeltaPatchCommand]:
        """Push a pre-computed dirty patch directly without scanning full buffers."""
        if dirty_rect[2] <= 0 or dirty_rect[3] <= 0:
            return None
        cmd = DeltaPatchCommand(
            layer_id=layer_id,
            target=target,
            dirty_rect=dirty_rect,
            before_patch=np.ascontiguousarray(before_patch.copy()),
            after_patch=np.ascontiguousarray(after_patch.copy()),
            description=description,
        )
        self.undo_stack.push(cmd)
        self.modified = True
        self.invalidate_composite()
        return cmd

    @contextmanager
    def edit_layer_context(
        self,
        layer: Optional[Layer] = None,
        target: str = "pixels",
        roi: Optional[tuple[int, int, int, int]] = None,
        description: str = "Edit",
    ) -> Iterator[None]:
        self.begin_layer_edit(layer=layer, target=target, roi=roi)
        try:
            yield
        except Exception:
            self.cancel_layer_edit()
            raise
        else:
            self.commit_layer_edit(description=description)

    # --- Layer Operations with Undo/Redo ---

    def add_layer(
        self,
        layer: Optional[Layer] = None,
        index: Optional[int] = None,
        name: Optional[str] = None,
        fill_rgba: tuple[int, int, int, int] = (0, 0, 0, 0),
    ) -> Layer:
        if layer is None:
            layer_name = name or f"Layer {len(self.layers) + 1}"
            new_layer = Layer(self.width, self.height, name=layer_name, fill_rgba=fill_rgba)
        else:
            new_layer = layer

        insert_idx = len(self.layers) if index is None else max(0, min(len(self.layers), int(index)))
        prev_active = self.active_layer_index

        def do(doc: DocumentSession) -> None:
            doc.layers.insert(insert_idx, new_layer)
            doc.active_layer_index = insert_idx

        def undo(doc: DocumentSession) -> None:
            doc.layers.pop(insert_idx)
            doc.active_layer_index = max(0, min(prev_active, len(doc.layers) - 1)) if doc.layers else -1

        do(self)
        self.undo_stack.push(
            CallableCommand(
                undo_fn=undo,
                redo_fn=do,
                description=f"Add Layer '{new_layer.name}'",
                estimated_bytes=int(new_layer.pixels.nbytes),
            )
        )
        self.modified = True
        self.invalidate_composite()
        return new_layer

    def remove_layer(self, index: Optional[int] = None) -> Layer:
        if len(self.layers) <= 1:
            raise ValueError("Cannot remove the last remaining layer in a document")
        idx = self.active_layer_index if index is None else int(index)
        if not (0 <= idx < len(self.layers)):
            raise IndexError(f"Layer index out of range: {idx}")

        removed = self.layers[idx]
        prev_active = self.active_layer_index

        def do(doc: DocumentSession) -> None:
            doc.layers.pop(idx)
            doc.active_layer_index = max(0, min(prev_active, len(doc.layers) - 1))

        def undo(doc: DocumentSession) -> None:
            doc.layers.insert(idx, removed)
            doc.active_layer_index = prev_active

        do(self)
        self.undo_stack.push(
            CallableCommand(
                undo_fn=undo,
                redo_fn=do,
                description=f"Delete Layer '{removed.name}'",
                estimated_bytes=int(removed.pixels.nbytes),
            )
        )
        self.modified = True
        self.invalidate_composite()
        return removed

    def duplicate_layer(self, index: Optional[int] = None) -> Layer:
        idx = self.active_layer_index if index is None else int(index)
        if not (0 <= idx < len(self.layers)):
            raise IndexError(f"Layer index out of range: {idx}")
        copy_layer = self.layers[idx].clone()
        return self.add_layer(layer=copy_layer, index=idx + 1)

    def move_layer(self, from_index: int, to_index: int) -> None:
        n = len(self.layers)
        if not (0 <= from_index < n) or not (0 <= to_index < n):
            raise IndexError(f"Layer reorder indices out of range: {from_index} -> {to_index}")
        if from_index == to_index:
            return
        prev_active = self.active_layer_index

        def do(doc: DocumentSession) -> None:
            item = doc.layers.pop(from_index)
            doc.layers.insert(to_index, item)
            doc.active_layer_index = to_index

        def undo(doc: DocumentSession) -> None:
            item = doc.layers.pop(to_index)
            doc.layers.insert(from_index, item)
            doc.active_layer_index = prev_active

        do(self)
        self.undo_stack.push(
            CallableCommand(undo_fn=undo, redo_fn=do, description="Reorder Layers")
        )
        self.modified = True
        self.invalidate_composite()

    def merge_down(self, index: Optional[int] = None) -> Layer:
        idx = self.active_layer_index if index is None else int(index)
        if idx <= 0 or idx >= len(self.layers):
            raise ValueError(f"Cannot merge down layer at index {idx}")

        upper = self.layers[idx]
        lower = self.layers[idx - 1]
        prev_active = self.active_layer_index

        # Render into canvas-sized merged buffer if sizes/offsets differ, else direct compositing
        merged_x0 = min(lower.offset_x, upper.offset_x)
        merged_y0 = min(lower.offset_y, upper.offset_y)
        merged_x1 = max(lower.offset_x + lower.width, upper.offset_x + upper.width)
        merged_y1 = max(lower.offset_y + lower.height, upper.offset_y + upper.height)
        mw = merged_x1 - merged_x0
        mh = merged_y1 - merged_y0

        merged_pixels = np.zeros((mh, mw, 4), dtype=np.uint8)
        engine = get_engine()
        lower_mask = lower.mask.buffer if (lower.mask is not None and lower.mask.enabled) else None
        upper_mask = upper.mask.buffer if (upper.mask is not None and upper.mask.enabled) else None

        if lower.visible and lower.opacity > 0.0:
            engine.composite_layer(
                dst=merged_pixels,
                src=lower.pixels,
                mask=lower_mask,
                offset_x=lower.offset_x - merged_x0,
                offset_y=lower.offset_y - merged_y0,
                opacity=lower.opacity,
                blend_mode=int(BlendMode.NORMAL),
            )
        if upper.visible and upper.opacity > 0.0:
            engine.composite_layer(
                dst=merged_pixels,
                src=upper.pixels,
                mask=upper_mask,
                offset_x=upper.offset_x - merged_x0,
                offset_y=upper.offset_y - merged_y0,
                opacity=upper.opacity,
                blend_mode=int(upper.blend_mode),
            )

        merged_layer = Layer(
            width=mw,
            height=mh,
            name=lower.name,
            pixels=merged_pixels,
            offset_x=merged_x0,
            offset_y=merged_y0,
            opacity=1.0,
            blend_mode=lower.blend_mode,
            visible=lower.visible,
            locked=lower.locked,
            clipping_mask=lower.clipping_mask,
        )

        def do(doc: DocumentSession) -> None:
            doc.layers[idx - 1] = merged_layer
            doc.layers.pop(idx)
            doc.active_layer_index = idx - 1

        def undo(doc: DocumentSession) -> None:
            doc.layers[idx - 1] = lower
            doc.layers.insert(idx, upper)
            doc.active_layer_index = prev_active

        do(self)
        self.undo_stack.push(
            CallableCommand(
                undo_fn=undo,
                redo_fn=do,
                description=f"Merge Down '{upper.name}'",
                estimated_bytes=int(lower.pixels.nbytes + upper.pixels.nbytes),
            )
        )
        self.modified = True
        self.invalidate_composite()
        return merged_layer

    def _record_layer_attr(self, index: int, attr: str, new_value, description: str) -> None:
        layer = self.layers[index]
        old_value = getattr(layer, attr)
        if old_value == new_value:
            return

        def do(doc: DocumentSession) -> None:
            setattr(layer, attr, new_value)

        def undo(doc: DocumentSession) -> None:
            setattr(layer, attr, old_value)

        do(self)
        self.undo_stack.push(CallableCommand(undo_fn=undo, redo_fn=do, description=description))
        self.modified = True
        self.invalidate_composite()

    def set_layer_opacity(self, index: int, opacity: float) -> None:
        clamped = max(0.0, min(1.0, float(opacity)))
        self._record_layer_attr(index, "opacity", clamped, "Change Opacity")

    def set_layer_blend_mode(self, index: int, blend_mode: BlendMode) -> None:
        self._record_layer_attr(index, "blend_mode", BlendMode(blend_mode), "Change Blend Mode")

    def set_layer_visible(self, index: int, visible: bool) -> None:
        self._record_layer_attr(index, "visible", bool(visible), "Toggle Visibility")

    def set_layer_locked(self, index: int, locked: bool) -> None:
        self._record_layer_attr(index, "locked", bool(locked), "Toggle Lock")

    def set_layer_clipping_mask(self, index: int, clipping: bool) -> None:
        self._record_layer_attr(index, "clipping_mask", bool(clipping), "Toggle Clipping Mask")

    def rename_layer(self, index: int, new_name: str) -> None:
        self._record_layer_attr(index, "name", str(new_name), "Rename Layer")

    def move_layer_offset(
        self,
        index: int,
        new_offset_x: int,
        new_offset_y: int,
        description: str = "Move Layer",
    ) -> None:
        layer = self.layers[index]
        old_x, old_y = layer.offset_x, layer.offset_y
        nx, ny = int(new_offset_x), int(new_offset_y)
        if old_x == nx and old_y == ny:
            return

        def do(doc: DocumentSession) -> None:
            layer.offset_x = nx
            layer.offset_y = ny

        def undo(doc: DocumentSession) -> None:
            layer.offset_x = old_x
            layer.offset_y = old_y

        do(self)
        self.undo_stack.push(CallableCommand(undo_fn=undo, redo_fn=do, description=description))
        self.modified = True
        self.invalidate_composite()

    def center_active_layer(
        self,
        horizontal: bool = True,
        vertical: bool = True,
        use_content_bounds: bool = True,
    ) -> None:
        layer = self.active_layer
        if layer is None:
            return
        old_x, old_y = layer.offset_x, layer.offset_y
        new_x, new_y = layer.center_on_canvas(
            self.width,
            self.height,
            horizontal=horizontal,
            vertical=vertical,
            use_content_bounds=use_content_bounds,
        )
        # Restore and route through undoable move_layer_offset
        layer.offset_x, layer.offset_y = old_x, old_y
        self.move_layer_offset(self.active_layer_index, new_x, new_y, description="Center Layer")

    def trim_transparent(self, alpha_threshold: int = 0) -> bool:
        """Crop the canvas tightly around non-transparent composited pixels."""
        composited = self.composite()
        bounds = get_engine().find_alpha_bounds(composited, alpha_threshold=alpha_threshold)
        if bounds is None:
            return False
        bx, by, bw, bh = bounds
        if bx == 0 and by == 0 and bw == self.width and bh == self.height:
            return False

        old_w, old_h = self.width, self.height
        old_offsets = [(layer.offset_x, layer.offset_y) for layer in self.layers]
        old_sel = self.selection_mask

        def do(doc: DocumentSession) -> None:
            doc.width = bw
            doc.height = bh
            for layer, (ox, oy) in zip(doc.layers, old_offsets):
                layer.offset_x = ox - bx
                layer.offset_y = oy - by
            doc.selection_mask = None
            doc._composited_cache = None
            doc._clip_alpha_scratch = None

        def undo(doc: DocumentSession) -> None:
            doc.width = old_w
            doc.height = old_h
            for layer, (ox, oy) in zip(doc.layers, old_offsets):
                layer.offset_x = ox
                layer.offset_y = oy
            doc.selection_mask = old_sel
            doc._composited_cache = None
            doc._clip_alpha_scratch = None

        do(self)
        self.undo_stack.push(CallableCommand(undo_fn=undo, redo_fn=do, description="Trim Transparent"))
        self.modified = True
        self.invalidate_composite()
        return True

    def undo(self) -> Optional[str]:
        desc = self.undo_stack.undo(self)
        if desc is not None:
            self.modified = True
            self.invalidate_composite()
        return desc

    def redo(self) -> Optional[str]:
        desc = self.undo_stack.redo(self)
        if desc is not None:
            self.modified = True
            self.invalidate_composite()
        return desc


class DocumentTabManager:
    """Manages multiple open DocumentSession tabs with zero-overhead inactive tabs."""

    def __init__(self) -> None:
        self.sessions: list[DocumentSession] = []
        self.active_index: int = -1
        self._untitled_counter: int = 0

    @property
    def active_document(self) -> Optional[DocumentSession]:
        if 0 <= self.active_index < len(self.sessions):
            return self.sessions[self.active_index]
        return None

    def get_active_document(self) -> Optional[DocumentSession]:
        return self.active_document

    def _next_untitled_title(self) -> str:
        self._untitled_counter += 1
        return f"Untitled-{self._untitled_counter}"

    def new_document(
        self,
        width: int = 800,
        height: int = 600,
        title: Optional[str] = None,
        initial_fill: tuple[int, int, int, int] = (0, 0, 0, 0),
        activate: bool = True,
    ) -> DocumentSession:
        doc = DocumentSession(
            width=width,
            height=height,
            title=title or self._next_untitled_title(),
            initial_fill=initial_fill,
            is_active_tab=False,
        )
        self.add_document(doc, activate=activate)
        return doc

    def new_document_from_rgba(
        self,
        rgba_buffer: np.ndarray,
        title: Optional[str] = None,
        activate: bool = True,
    ) -> DocumentSession:
        layer = Layer.from_rgba(rgba_buffer, name="Layer 1", copy=True)
        doc = DocumentSession(
            width=layer.width,
            height=layer.height,
            title=title or self._next_untitled_title(),
            layers=[layer],
            is_active_tab=False,
        )
        self.add_document(doc, activate=activate)
        return doc

    def add_document(self, doc: DocumentSession, activate: bool = True) -> int:
        doc.set_tab_active(False)
        self.sessions.append(doc)
        new_idx = len(self.sessions) - 1
        if activate or self.active_index == -1:
            self.set_active_tab(new_idx)
        return new_idx

    def set_active_tab(self, index: int) -> DocumentSession:
        if not (0 <= index < len(self.sessions)):
            raise IndexError(f"Tab index out of range: {index}")
        if 0 <= self.active_index < len(self.sessions) and self.active_index != index:
            self.sessions[self.active_index].set_tab_active(False)
        self.active_index = int(index)
        active_doc = self.sessions[self.active_index]
        active_doc.set_tab_active(True)
        return active_doc

    def close_tab(self, index: int) -> Optional[DocumentSession]:
        if not (0 <= index < len(self.sessions)):
            raise IndexError(f"Tab index out of range: {index}")
        closed = self.sessions.pop(index)
        closed.set_tab_active(False)

        if not self.sessions:
            self.active_index = -1
            return closed

        if index == self.active_index:
            new_idx = min(index, len(self.sessions) - 1)
            self.active_index = new_idx
            self.sessions[new_idx].set_tab_active(True)
        elif index < self.active_index:
            self.active_index -= 1
        return closed
