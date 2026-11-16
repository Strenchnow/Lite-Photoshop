"""Base tool interface, shared ToolContext, ToolType enum, and modifier helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np

if TYPE_CHECKING:
    from app.core.document import DocumentSession, ViewportTransform
    from app.core.layer import Layer


MOD_NONE: int = 0
MOD_SHIFT: int = 1
MOD_CTRL: int = 2
MOD_ALT: int = 4

_QT_SHIFT_MODIFIER = 0x02000000
_QT_CTRL_MODIFIER = 0x04000000
_QT_ALT_MODIFIER = 0x08000000


def _to_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    raw = getattr(value, "value", value)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def extract_modifiers(qt_mods: Any) -> int:
    """Convert Qt KeyboardModifiers or plain bitmask int into MOD_* flags."""
    raw = _to_int(qt_mods, default=0)
    mods = raw & (MOD_SHIFT | MOD_CTRL | MOD_ALT)
    if raw & _QT_SHIFT_MODIFIER:
        mods |= MOD_SHIFT
    if raw & _QT_CTRL_MODIFIER:
        mods |= MOD_CTRL
    if raw & _QT_ALT_MODIFIER:
        mods |= MOD_ALT
    return mods


def extract_button(button: Any) -> int:
    """Normalize a Qt MouseButton/MouseButtons value or int to an integer bitmask."""
    return _to_int(button, default=0)


def extract_key(key: Any) -> int:
    """Normalize a Qt Key enum or plain int to an integer key code."""
    return _to_int(key, default=0)


class ToolType(str, Enum):
    """Canonical string identifiers for all editor tools."""

    MOVE = "move"
    HAND = "hand"
    BRUSH = "brush"
    ERASER = "eraser"
    SMUDGE = "smudge"
    BLUR_SHARPEN = "blur_sharpen"
    MARQUEE = "marquee"
    LASSO = "lasso"
    MAGIC_WAND = "magic_wand"
    CROP = "crop"
    TRANSFORM = "transform"
    EYEDROPPER = "eyedropper"
    BUCKET_FILL = "bucket_fill"
    GRADIENT = "gradient"
    TEXT = "text"


CROP_ASPECT_PRESETS: dict[str, Optional[tuple[float, float]]] = {
    "free": None,
    "1:1": (1.0, 1.0),
    "4:3": (4.0, 3.0),
    "3:4": (3.0, 4.0),
    "16:9": (16.0, 9.0),
    "9:16": (9.0, 16.0),
    "3:2": (3.0, 2.0),
    "2:3": (2.0, 3.0),
}

_CROP_SWAP_MAP: dict[str, str] = {
    "4:3": "3:4",
    "3:4": "4:3",
    "16:9": "9:16",
    "9:16": "16:9",
    "3:2": "2:3",
    "2:3": "3:2",
}


@dataclass
class ToolContext:
    """Shared mutable tool settings and UI callbacks across all tools."""

    fg_color: tuple[int, int, int, int] = (255, 255, 255, 255)
    bg_color: tuple[int, int, int, int] = (0, 0, 0, 255)
    brush_radius: float = 12.0
    brush_hardness: float = 0.8
    brush_opacity: float = 1.0
    brush_flow: float = 1.0
    brush_spacing: float = 0.2
    smudge_strength: float = 0.7
    filter_mode: int = 0
    filter_kernel_size: int = 3
    filter_strength: float = 0.6
    wand_tolerance: int = 32
    wand_contiguous: bool = True
    fill_tolerance: int = 32
    fill_contiguous: bool = True
    crop_aspect_preset: str = "free"
    crop_aspect_ratio: Optional[tuple[float, float]] = None
    font_family: str = "Segoe UI"
    font_size: int = 28
    font_bold: bool = False
    text_content: str = "Text"
    edit_mask: bool = False
    snap_guides: list[tuple[str, float]] = field(default_factory=list)
    on_color_picked: Optional[Callable[[tuple[int, int, int, int]], None]] = None
    on_canvas_invalidated: Optional[Callable[[], None]] = None
    on_document_resized: Optional[Callable[[], None]] = None

    def set_crop_preset(self, preset: str) -> None:
        key = str(preset).strip().lower()
        if key not in CROP_ASPECT_PRESETS:
            key = "free"
        self.crop_aspect_preset = key
        self.crop_aspect_ratio = CROP_ASPECT_PRESETS[key]

    def swap_crop_orientation(self) -> str:
        if self.crop_aspect_preset in _CROP_SWAP_MAP:
            swapped = _CROP_SWAP_MAP[self.crop_aspect_preset]
            self.set_crop_preset(swapped)
        elif self.crop_aspect_ratio is not None:
            rw, rh = self.crop_aspect_ratio
            self.crop_aspect_ratio = (rh, rw)
        return self.crop_aspect_preset

    def notify_color_picked(self, color: tuple[int, int, int, int]) -> None:
        if self.on_color_picked is not None:
            self.on_color_picked(color)

    def notify_canvas_invalidated(self) -> None:
        if self.on_canvas_invalidated is not None:
            self.on_canvas_invalidated()

    def notify_document_resized(self) -> None:
        if self.on_document_resized is not None:
            self.on_document_resized()


def align_selection_mask_to_layer(
    doc: "DocumentSession",
    layer: "Layer",
) -> Optional[np.ndarray]:
    """Map canvas-space doc.selection_mask to layer-local coordinates if needed."""
    sel = doc.selection_mask
    if sel is None:
        return None
    if (
        layer.offset_x == 0
        and layer.offset_y == 0
        and layer.width == doc.width
        and layer.height == doc.height
    ):
        return sel

    aligned = np.zeros((layer.height, layer.width), dtype=np.uint8)
    cx0 = max(0, layer.offset_x)
    cy0 = max(0, layer.offset_y)
    cx1 = min(doc.width, layer.offset_x + layer.width)
    cy1 = min(doc.height, layer.offset_y + layer.height)
    if cx1 <= cx0 or cy1 <= cy0:
        return aligned

    lx0 = cx0 - layer.offset_x
    ly0 = cy0 - layer.offset_y
    lx1 = lx0 + (cx1 - cx0)
    ly1 = ly0 + (cy1 - cy0)
    aligned[ly0:ly1, lx0:lx1] = sel[cy0:cy1, cx0:cx1]
    return aligned


def merge_rect(
    r1: Optional[tuple[int, int, int, int]],
    r2: Optional[tuple[int, int, int, int]],
) -> Optional[tuple[int, int, int, int]]:
    """Return the bounding union of two (x, y, w, h) rectangles."""
    if r1 is None or r1[2] <= 0 or r1[3] <= 0:
        return r2 if (r2 is not None and r2[2] > 0 and r2[3] > 0) else None
    if r2 is None or r2[2] <= 0 or r2[3] <= 0:
        return r1
    x0 = min(r1[0], r2[0])
    y0 = min(r1[1], r2[1])
    x1 = max(r1[0] + r1[2], r2[0] + r2[2])
    y1 = max(r1[1] + r1[3], r2[1] + r2[3])
    return (x0, y0, x1 - x0, y1 - y0)


class BaseTool:
    """Abstract base interface for all canvas interaction tools."""

    tool_id: str = ""
    name: str = ""
    cursor_hint: str = "arrow"

    def __init__(self, context: ToolContext) -> None:
        self.context = context

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
        return None

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
        return None

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
        return None

    def key_press(
        self,
        doc: "DocumentSession",
        key: int,
        modifiers: int = 0,
    ) -> bool:
        return False

    def draw_overlay(
        self,
        painter: Any,
        doc: "DocumentSession",
        viewport: "ViewportTransform",
    ) -> None:
        return None

    def on_activate(self, doc: Optional["DocumentSession"] = None) -> None:
        return None

    def on_deactivate(self, doc: Optional["DocumentSession"] = None) -> None:
        return None

    # camelCase aliases matching SPEC.md
    def mousePress(
        self,
        doc: "DocumentSession",
        cx: float,
        cy: float,
        button: int = 1,
        modifiers: int = 0,
        sx: float = 0.0,
        sy: float = 0.0,
    ) -> None:
        self.mouse_press(doc, cx, cy, button=button, modifiers=modifiers, sx=sx, sy=sy)

    def mouseMove(
        self,
        doc: "DocumentSession",
        cx: float,
        cy: float,
        buttons: int = 0,
        modifiers: int = 0,
        sx: float = 0.0,
        sy: float = 0.0,
    ) -> None:
        self.mouse_move(doc, cx, cy, buttons=buttons, modifiers=modifiers, sx=sx, sy=sy)

    def mouseRelease(
        self,
        doc: "DocumentSession",
        cx: float,
        cy: float,
        button: int = 1,
        modifiers: int = 0,
        sx: float = 0.0,
        sy: float = 0.0,
    ) -> None:
        self.mouse_release(doc, cx, cy, button=button, modifiers=modifiers, sx=sx, sy=sy)

    def keyPress(
        self,
        doc: "DocumentSession",
        key: int,
        modifiers: int = 0,
    ) -> bool:
        return self.key_press(doc, key, modifiers=modifiers)
