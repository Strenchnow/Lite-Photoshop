"""Layer and LayerMask models backed by contiguous uint8 NumPy buffers."""

from __future__ import annotations

import uuid
from enum import IntEnum
from typing import Optional

import numpy as np

from app.core.c_bridge import (
    BLEND_ADD,
    BLEND_MULTIPLY,
    BLEND_NORMAL,
    BLEND_OVERLAY,
    BLEND_SCREEN,
    get_engine,
)


class BlendMode(IntEnum):
    """Supported C-accelerated layer blend modes."""

    NORMAL = BLEND_NORMAL
    MULTIPLY = BLEND_MULTIPLY
    SCREEN = BLEND_SCREEN
    OVERLAY = BLEND_OVERLAY
    ADD = BLEND_ADD

    @property
    def label(self) -> str:
        return BLEND_MODE_LABELS[self]

    @classmethod
    def from_label(cls, label: str) -> "BlendMode":
        normalized = label.strip().lower()
        for mode, text in BLEND_MODE_LABELS.items():
            if text.lower() == normalized:
                return mode
        raise ValueError(f"Unknown blend mode: {label!r}")


BLEND_MODE_LABELS: dict[BlendMode, str] = {
    BlendMode.NORMAL: "Normal",
    BlendMode.MULTIPLY: "Multiply",
    BlendMode.SCREEN: "Screen",
    BlendMode.OVERLAY: "Overlay",
    BlendMode.ADD: "Add",
}


def _clip_rect(rect: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x, y, w, h = rect
    x0 = max(0, int(x))
    y0 = max(0, int(y))
    x1 = min(width, int(x) + int(w))
    y1 = min(height, int(y) + int(h))
    if x1 <= x0 or y1 <= y0:
        return (x0, y0, 0, 0)
    return (x0, y0, x1 - x0, y1 - y0)


def _ensure_contiguous_rgba(pixels: np.ndarray) -> np.ndarray:
    arr = np.asarray(pixels, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 4:
        raise ValueError(f"Layer pixels must have shape (H, W, 4), got {arr.shape}")
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr, dtype=np.uint8)
    return arr


def _ensure_contiguous_mask(buffer: np.ndarray) -> np.ndarray:
    arr = np.asarray(buffer, dtype=np.uint8)
    if arr.ndim != 2:
        raise ValueError(f"LayerMask buffer must have shape (H, W), got {arr.shape}")
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr, dtype=np.uint8)
    return arr


class LayerMask:
    """Single-channel uint8 grayscale raster mask attached to a Layer."""

    def __init__(
        self,
        width: int,
        height: int,
        fill_value: int = 255,
        buffer: Optional[np.ndarray] = None,
        enabled: bool = True,
    ) -> None:
        if buffer is not None:
            buf = _ensure_contiguous_mask(buffer)
            self.height, self.width = int(buf.shape[0]), int(buf.shape[1])
            self.buffer = buf
        else:
            if width <= 0 or height <= 0:
                raise ValueError(f"LayerMask dimensions must be positive, got ({width}, {height})")
            self.width = int(width)
            self.height = int(height)
            self.buffer = np.full((self.height, self.width), fill_value & 0xFF, dtype=np.uint8)
        self.enabled = bool(enabled)

    def fill(self, value: int) -> None:
        self.buffer.fill(value & 0xFF)

    def invert(self) -> None:
        np.subtract(255, self.buffer, out=self.buffer, dtype=np.uint8)

    def clone(self) -> "LayerMask":
        return LayerMask(
            width=self.width,
            height=self.height,
            buffer=self.buffer.copy(),
            enabled=self.enabled,
        )

    def get_patch(self, rect: tuple[int, int, int, int]) -> np.ndarray:
        x, y, w, h = _clip_rect(rect, self.width, self.height)
        if w <= 0 or h <= 0:
            return np.empty((0, 0), dtype=np.uint8)
        return np.ascontiguousarray(self.buffer[y : y + h, x : x + w].copy())

    def set_patch(self, rect: tuple[int, int, int, int], patch: np.ndarray) -> None:
        x, y, w, h = _clip_rect(rect, self.width, self.height)
        if w <= 0 or h <= 0:
            return
        if patch.shape != (h, w):
            raise ValueError(f"Mask patch shape mismatch: expected ({h}, {w}), got {patch.shape}")
        self.buffer[y : y + h, x : x + w] = patch

    def resize(self, new_w: int, new_h: int) -> None:
        if new_w <= 0 or new_h <= 0:
            raise ValueError(f"Mask target size must be positive, got ({new_w}, {new_h})")
        if new_w == self.width and new_h == self.height:
            return
        rgba = np.empty((self.height, self.width, 4), dtype=np.uint8)
        rgba[:, :, 0] = self.buffer
        rgba[:, :, 1] = self.buffer
        rgba[:, :, 2] = self.buffer
        rgba[:, :, 3] = 255
        resized_rgba = get_engine().bilinear_resize(rgba, int(new_w), int(new_h))
        self.buffer = np.ascontiguousarray(resized_rgba[:, :, 0].copy(), dtype=np.uint8)
        self.width = int(new_w)
        self.height = int(new_h)


class Layer:
    """Raster RGBA layer with transform offset, opacity, blend mode, and optional mask."""

    def __init__(
        self,
        width: int,
        height: int,
        name: str = "Layer",
        pixels: Optional[np.ndarray] = None,
        fill_rgba: Optional[tuple[int, int, int, int]] = None,
        offset_x: int = 0,
        offset_y: int = 0,
        opacity: float = 1.0,
        blend_mode: BlendMode = BlendMode.NORMAL,
        visible: bool = True,
        locked: bool = False,
        clipping_mask: bool = False,
        mask: Optional[LayerMask] = None,
        layer_id: Optional[str] = None,
    ) -> None:
        if pixels is not None:
            self.pixels = _ensure_contiguous_rgba(pixels)
        else:
            if width <= 0 or height <= 0:
                raise ValueError(f"Layer dimensions must be positive, got ({width}, {height})")
            self.pixels = np.zeros((int(height), int(width), 4), dtype=np.uint8)
            if fill_rgba is not None and fill_rgba != (0, 0, 0, 0):
                r, g, b, a = fill_rgba
                get_engine().clear_buffer(self.pixels, r, g, b, a)

        self.id = layer_id or uuid.uuid4().hex
        self.name = str(name)
        self.offset_x = int(offset_x)
        self.offset_y = int(offset_y)
        self.opacity = max(0.0, min(1.0, float(opacity)))
        self.blend_mode = BlendMode(blend_mode)
        self.visible = bool(visible)
        self.locked = bool(locked)
        self.clipping_mask = bool(clipping_mask)
        self.mask = mask

    @classmethod
    def from_rgba(
        cls,
        pixels: np.ndarray,
        name: str = "Layer",
        copy: bool = True,
        **kwargs,
    ) -> "Layer":
        buf = pixels.copy() if copy else pixels
        buf = _ensure_contiguous_rgba(buf)
        h, w = int(buf.shape[0]), int(buf.shape[1])
        return cls(width=w, height=h, name=name, pixels=buf, **kwargs)

    @property
    def width(self) -> int:
        return int(self.pixels.shape[1])

    @property
    def height(self) -> int:
        return int(self.pixels.shape[0])

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        """Return layer bounding box in canvas coordinates: (offset_x, offset_y, width, height)."""
        return (self.offset_x, self.offset_y, self.width, self.height)

    def create_mask(self, fill_value: int = 255) -> LayerMask:
        self.mask = LayerMask(self.width, self.height, fill_value=fill_value)
        return self.mask

    def delete_mask(self, apply_before_delete: bool = False) -> None:
        if self.mask is None:
            return
        if apply_before_delete and self.mask.enabled:
            # Multiply layer alpha by mask alpha in-place
            alpha = self.pixels[:, :, 3].astype(np.uint16)
            masked = (alpha * self.mask.buffer.astype(np.uint16) + 127) // 255
            self.pixels[:, :, 3] = masked.astype(np.uint8)
        self.mask = None

    def clone(self, new_name: Optional[str] = None) -> "Layer":
        return Layer(
            width=self.width,
            height=self.height,
            name=new_name or f"{self.name} Copy",
            pixels=self.pixels.copy(),
            offset_x=self.offset_x,
            offset_y=self.offset_y,
            opacity=self.opacity,
            blend_mode=self.blend_mode,
            visible=self.visible,
            locked=self.locked,
            clipping_mask=self.clipping_mask,
            mask=self.mask.clone() if self.mask is not None else None,
        )

    def clear(self, r: int = 0, g: int = 0, b: int = 0, a: int = 0) -> None:
        get_engine().clear_buffer(self.pixels, r, g, b, a)

    def get_patch(self, rect: tuple[int, int, int, int]) -> np.ndarray:
        x, y, w, h = _clip_rect(rect, self.width, self.height)
        if w <= 0 or h <= 0:
            return np.empty((0, 0, 4), dtype=np.uint8)
        return np.ascontiguousarray(self.pixels[y : y + h, x : x + w].copy())

    def set_patch(self, rect: tuple[int, int, int, int], patch: np.ndarray) -> None:
        x, y, w, h = _clip_rect(rect, self.width, self.height)
        if w <= 0 or h <= 0:
            return
        if patch.shape != (h, w, 4):
            raise ValueError(f"Layer patch shape mismatch: expected ({h}, {w}, 4), got {patch.shape}")
        self.pixels[y : y + h, x : x + w] = patch

    def get_non_transparent_bounds(self, alpha_threshold: int = 0) -> Optional[tuple[int, int, int, int]]:
        """Return tight non-transparent bounding box in canvas coordinates, or None if empty."""
        local_rect = get_engine().find_alpha_bounds(self.pixels, alpha_threshold=alpha_threshold)
        if local_rect is None:
            return None
        lx, ly, lw, lh = local_rect
        return (self.offset_x + lx, self.offset_y + ly, lw, lh)

    def center_on_canvas(
        self,
        canvas_w: int,
        canvas_h: int,
        horizontal: bool = True,
        vertical: bool = True,
        use_content_bounds: bool = True,
    ) -> tuple[int, int]:
        """Center layer (or its opaque content) on the canvas and return the new (offset_x, offset_y)."""
        bounds = self.get_non_transparent_bounds() if use_content_bounds else None
        if bounds is None:
            bx, by, bw, bh = self.bounds
        else:
            bx, by, bw, bh = bounds

        if horizontal:
            target_x = (int(canvas_w) - bw) // 2
            self.offset_x += target_x - bx
        if vertical:
            target_y = (int(canvas_h) - bh) // 2
            self.offset_y += target_y - by
        return (self.offset_x, self.offset_y)
