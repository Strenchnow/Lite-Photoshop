"""Grid Splitter and C-accelerated Smart Alpha Connected-Component Sprite Splitter."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from app.core.c_bridge import ComponentBox, get_engine
from app.core.document import DocumentSession, DocumentTabManager


def _ensure_rgba_u8(rgba: np.ndarray) -> np.ndarray:
    if not isinstance(rgba, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(rgba).__name__}")
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError(f"Expected RGBA buffer shape (H, W, 4), got {rgba.shape}")
    if rgba.dtype != np.uint8 or not rgba.flags["C_CONTIGUOUS"]:
        return np.ascontiguousarray(rgba, dtype=np.uint8)
    return rgba


def _slice_pixels(item: object) -> np.ndarray:
    if isinstance(item, np.ndarray):
        return _ensure_rgba_u8(item)
    if isinstance(item, (tuple, list)) and len(item) >= 2:
        return _ensure_rgba_u8(item[1])
    raise TypeError(
        "Each slice must be an RGBA ndarray or a (box, rgba) tuple; "
        f"got {type(item).__name__}"
    )


def split_grid(
    rgba: np.ndarray,
    rows: int = 2,
    cols: int = 2,
    tile_w: int = 0,
    tile_h: int = 0,
) -> list[tuple[tuple[int, int, int, int], np.ndarray]]:
    """Split an RGBA buffer into uniform rows x cols cells or fixed tile_w x tile_h tiles."""
    buf = _ensure_rgba_u8(rgba)
    img_h, img_w = int(buf.shape[0]), int(buf.shape[1])
    if img_w <= 0 or img_h <= 0:
        return []

    if tile_w < 0 or tile_h < 0:
        raise ValueError(f"Tile dimensions must be non-negative, got ({tile_w}, {tile_h})")

    slices: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
    if tile_w > 0 or tile_h > 0:
        tw = int(tile_w) if tile_w > 0 else img_w
        th = int(tile_h) if tile_h > 0 else img_h
        for y0 in range(0, img_h, th):
            h = min(th, img_h - y0)
            for x0 in range(0, img_w, tw):
                w = min(tw, img_w - x0)
                crop = np.ascontiguousarray(buf[y0 : y0 + h, x0 : x0 + w].copy(), dtype=np.uint8)
                slices.append(((x0, y0, w, h), crop))
        return slices

    if rows < 1 or cols < 1:
        raise ValueError(f"rows and cols must be >= 1, got rows={rows}, cols={cols}")

    for r in range(int(rows)):
        y0 = (r * img_h) // int(rows)
        y1 = ((r + 1) * img_h) // int(rows)
        if y1 <= y0:
            continue
        for c in range(int(cols)):
            x0 = (c * img_w) // int(cols)
            x1 = ((c + 1) * img_w) // int(cols)
            if x1 <= x0:
                continue
            w = x1 - x0
            h = y1 - y0
            crop = np.ascontiguousarray(buf[y0:y1, x0:x1].copy(), dtype=np.uint8)
            slices.append(((x0, y0, w, h), crop))
    return slices


def detect_alpha_sprites(
    rgba: np.ndarray,
    alpha_threshold: int = 10,
    min_area: int = 16,
    padding: int = 2,
) -> list[ComponentBox]:
    """Detect non-transparent connected-component sprite bounding boxes via the C engine."""
    buf = _ensure_rgba_u8(rgba)
    boxes, _ = get_engine().connected_components_alpha(
        buf,
        alpha_threshold=int(alpha_threshold),
        min_area=int(min_area),
        padding=int(padding),
        return_labels=False,
    )
    return boxes


def extract_alpha_sprites(
    rgba: np.ndarray,
    boxes: list[ComponentBox],
) -> list[tuple[ComponentBox, np.ndarray]]:
    """Crop each detected ComponentBox out of `rgba` as a contiguous RGBA patch."""
    buf = _ensure_rgba_u8(rgba)
    img_h, img_w = int(buf.shape[0]), int(buf.shape[1])
    extracted: list[tuple[ComponentBox, np.ndarray]] = []
    for box in boxes:
        x0 = max(0, int(box.x))
        y0 = max(0, int(box.y))
        x1 = min(img_w, int(box.x) + int(box.width))
        y1 = min(img_h, int(box.y) + int(box.height))
        if x1 <= x0 or y1 <= y0:
            continue
        crop = np.ascontiguousarray(buf[y0:y1, x0:x1].copy(), dtype=np.uint8)
        extracted.append((box, crop))
    return extracted


def export_slices_to_folder(
    slices: Iterable[object],
    output_dir: Path,
    prefix: str = "sprite",
) -> list[Path]:
    """Export grid or alpha slices as RGBA PNG files into `output_dir`."""
    slice_list = list(slices)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not slice_list:
        return []

    digits = max(3, len(str(len(slice_list))))
    clean_prefix = (prefix or "sprite").strip() or "sprite"
    saved_paths: list[Path] = []

    for idx, item in enumerate(slice_list, start=1):
        rgba = _slice_pixels(item)
        out_path = out_dir / f"{clean_prefix}_{idx:0{digits}d}.png"
        Image.fromarray(rgba, mode="RGBA").save(out_path, format="PNG")
        saved_paths.append(out_path)
    return saved_paths


def open_slices_as_tabs(
    slices: Iterable[object],
    tab_manager: DocumentTabManager,
    prefix: str = "Sprite",
) -> list[DocumentSession]:
    """Open each extracted slice as a new DocumentSession tab in `tab_manager`."""
    slice_list = list(slices)
    if not slice_list:
        return []

    clean_prefix = (prefix or "Sprite").strip() or "Sprite"
    opened: list[DocumentSession] = []
    total = len(slice_list)
    for idx, item in enumerate(slice_list, start=1):
        rgba = _slice_pixels(item)
        title = f"{clean_prefix} {idx}"
        doc = tab_manager.new_document_from_rgba(
            rgba,
            title=title,
            activate=(idx == total),
        )
        opened.append(doc)
    return opened
