"""Zero-copy ctypes bridge to the C pixel engine (core_engine.dll)."""

from __future__ import annotations

import ctypes
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


BLEND_NORMAL = 0
BLEND_MULTIPLY = 1
BLEND_SCREEN = 2
BLEND_OVERLAY = 3
BLEND_ADD = 4

FILTER_MODE_BLUR = 0
FILTER_MODE_SHARPEN = 1

U8_PTR = ctypes.POINTER(ctypes.c_uint8)
I32_PTR = ctypes.POINTER(ctypes.c_int32)
INT_PTR = ctypes.POINTER(ctypes.c_int)


class CComponentBox(ctypes.Structure):
    """C-compatible bounding box struct returned by connected_components_alpha."""

    _fields_ = [
        ("x", ctypes.c_int),
        ("y", ctypes.c_int),
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("area", ctypes.c_int),
        ("label", ctypes.c_int),
    ]


@dataclass(frozen=True, slots=True)
class ComponentBox:
    """Detected non-transparent island bounding box."""

    x: int
    y: int
    width: int
    height: int
    area: int
    label: int


def _lib_filename() -> str:
    if sys.platform == "win32":
        return "core_engine.dll"
    if sys.platform == "darwin":
        return "libcore_engine.dylib"
    return "libcore_engine.so"


def _resolve_library_path(auto_compile: bool = True) -> Path:
    lib_name = _lib_filename()
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        meipass = Path(sys._MEIPASS)  # type: ignore[attr-defined]
        for candidate in (meipass / "c_core" / lib_name, meipass / lib_name):
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Bundled {lib_name} not found in PyInstaller _MEIPASS: {meipass}")

    project_root = Path(__file__).resolve().parents[2]
    dll_path = project_root / "c_core" / lib_name

    if auto_compile:
        from c_core.compile_dll import compile_core_dll

        return compile_core_dll(force=False)

    if not dll_path.exists():
        raise FileNotFoundError(f"Shared library not found: {dll_path}")
    return dll_path


def _validate_rgba(arr: np.ndarray, name: str = "buffer") -> tuple[int, int]:
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray, got {type(arr).__name__}")
    if arr.dtype != np.uint8:
        raise ValueError(f"{name} must have dtype uint8, got {arr.dtype}")
    if arr.ndim != 3 or arr.shape[2] != 4:
        raise ValueError(f"{name} must have shape (H, W, 4), got {arr.shape}")
    if not arr.flags["C_CONTIGUOUS"]:
        raise ValueError(f"{name} must be C-contiguous for zero-copy C access")
    return int(arr.shape[1]), int(arr.shape[0])


def _validate_mask_2d(
    arr: np.ndarray,
    expected_w: Optional[int] = None,
    expected_h: Optional[int] = None,
    name: str = "mask",
) -> tuple[int, int]:
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray, got {type(arr).__name__}")
    if arr.dtype != np.uint8:
        raise ValueError(f"{name} must have dtype uint8, got {arr.dtype}")
    if arr.ndim != 2:
        raise ValueError(f"{name} must have shape (H, W), got {arr.shape}")
    if not arr.flags["C_CONTIGUOUS"]:
        raise ValueError(f"{name} must be C-contiguous for zero-copy C access")
    h, w = int(arr.shape[0]), int(arr.shape[1])
    if expected_w is not None and expected_h is not None and (w != expected_w or h != expected_h):
        raise ValueError(f"{name} shape mismatch: expected ({expected_h}, {expected_w}), got ({h}, {w})")
    return w, h


def _opt_u8_ptr(
    arr: Optional[np.ndarray],
    expected_w: int,
    expected_h: int,
    name: str,
) -> ctypes.POINTER(ctypes.c_uint8):
    if arr is None:
        return ctypes.POINTER(ctypes.c_uint8)()
    _validate_mask_2d(arr, expected_w=expected_w, expected_h=expected_h, name=name)
    return arr.ctypes.data_as(U8_PTR)


def _dirty_rect_to_tuple(rect_arr: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    x, y, w, h = (int(rect_arr[0]), int(rect_arr[1]), int(rect_arr[2]), int(rect_arr[3]))
    if w <= 0 or h <= 0:
        return None
    return (x, y, w, h)


class CoreEngineBridge:
    """Wrapper around core_engine.dll exposing zero-copy NumPy operations."""

    REQUIRED_SYMBOLS = (
        "clear_buffer_rgba",
        "composite_layer",
        "extract_alpha",
        "brush_dab_rgba",
        "brush_stroke_rgba",
        "brush_stroke_mask",
        "smudge_stroke_rgba",
        "blur_sharpen_dab_rgba",
        "gaussian_blur_rgba",
        "chroma_key_rgba",
        "flood_fill_rgba",
        "magic_wand_select",
        "connected_components_alpha",
        "find_alpha_bounds",
        "bilinear_resize_rgba",
    )

    def __init__(self, dll_path: Optional[Path] = None, auto_compile: bool = True) -> None:
        self.dll_path = Path(dll_path) if dll_path is not None else _resolve_library_path(auto_compile=auto_compile)
        self.lib = ctypes.CDLL(str(self.dll_path))
        self._bind_signatures()

    def _bind_signatures(self) -> None:
        lib = self.lib
        for sym in self.REQUIRED_SYMBOLS:
            if not hasattr(lib, sym):
                raise AttributeError(f"Missing required C export '{sym}' in {self.dll_path}")

        lib.clear_buffer_rgba.argtypes = [
            U8_PTR, ctypes.c_int, ctypes.c_int,
            ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint8,
        ]
        lib.clear_buffer_rgba.restype = None

        lib.composite_layer.argtypes = [
            U8_PTR, ctypes.c_int, ctypes.c_int,
            U8_PTR, ctypes.c_int, ctypes.c_int,
            U8_PTR, U8_PTR,
            ctypes.c_int, ctypes.c_int,
            ctypes.c_float, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]
        lib.composite_layer.restype = None

        lib.extract_alpha.argtypes = [U8_PTR, U8_PTR, ctypes.c_int, ctypes.c_int]
        lib.extract_alpha.restype = None

        lib.brush_dab_rgba.argtypes = [
            U8_PTR, ctypes.c_int, ctypes.c_int, U8_PTR,
            ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
            ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint8,
            ctypes.c_float, ctypes.c_float, ctypes.c_int, INT_PTR,
        ]
        lib.brush_dab_rgba.restype = None

        lib.brush_stroke_rgba.argtypes = [
            U8_PTR, ctypes.c_int, ctypes.c_int, U8_PTR,
            ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
            ctypes.c_float, ctypes.c_float, ctypes.c_float,
            ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint8,
            ctypes.c_float, ctypes.c_float, ctypes.c_int, INT_PTR,
        ]
        lib.brush_stroke_rgba.restype = None

        lib.brush_stroke_mask.argtypes = [
            U8_PTR, ctypes.c_int, ctypes.c_int, U8_PTR,
            ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
            ctypes.c_float, ctypes.c_float, ctypes.c_float,
            ctypes.c_uint8, ctypes.c_float, ctypes.c_float, INT_PTR,
        ]
        lib.brush_stroke_mask.restype = None

        lib.smudge_stroke_rgba.argtypes = [
            U8_PTR, ctypes.c_int, ctypes.c_int, U8_PTR,
            ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
            ctypes.c_float, ctypes.c_float, ctypes.c_float, INT_PTR,
        ]
        lib.smudge_stroke_rgba.restype = None

        lib.blur_sharpen_dab_rgba.argtypes = [
            U8_PTR, ctypes.c_int, ctypes.c_int, U8_PTR,
            ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
            ctypes.c_float, ctypes.c_int, ctypes.c_int, INT_PTR,
        ]
        lib.blur_sharpen_dab_rgba.restype = None

        lib.gaussian_blur_rgba.argtypes = [U8_PTR, U8_PTR, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        lib.gaussian_blur_rgba.restype = None

        lib.chroma_key_rgba.argtypes = [
            U8_PTR, ctypes.c_int, ctypes.c_int,
            ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint8,
            ctypes.c_float, ctypes.c_float, ctypes.c_float,
        ]
        lib.chroma_key_rgba.restype = None

        lib.flood_fill_rgba.argtypes = [
            U8_PTR, ctypes.c_int, ctypes.c_int, U8_PTR,
            ctypes.c_int, ctypes.c_int,
            ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint8,
            ctypes.c_int, ctypes.c_int, INT_PTR,
        ]
        lib.flood_fill_rgba.restype = ctypes.c_int

        lib.magic_wand_select.argtypes = [
            U8_PTR, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            U8_PTR, INT_PTR,
        ]
        lib.magic_wand_select.restype = ctypes.c_int

        lib.connected_components_alpha.argtypes = [
            U8_PTR, ctypes.c_int, ctypes.c_int,
            ctypes.c_uint8, ctypes.c_int, ctypes.c_int,
            I32_PTR, ctypes.POINTER(CComponentBox), ctypes.c_int,
        ]
        lib.connected_components_alpha.restype = ctypes.c_int

        lib.find_alpha_bounds.argtypes = [
            U8_PTR, ctypes.c_int, ctypes.c_int, ctypes.c_uint8, INT_PTR,
        ]
        lib.find_alpha_bounds.restype = ctypes.c_int

        lib.bilinear_resize_rgba.argtypes = [
            U8_PTR, ctypes.c_int, ctypes.c_int,
            U8_PTR, ctypes.c_int, ctypes.c_int,
        ]
        lib.bilinear_resize_rgba.restype = None

    def clear_buffer(self, dst: np.ndarray, r: int = 0, g: int = 0, b: int = 0, a: int = 0) -> None:
        w, h = _validate_rgba(dst, "dst")
        self.lib.clear_buffer_rgba(
            dst.ctypes.data_as(U8_PTR), w, h,
            r & 0xFF, g & 0xFF, b & 0xFF, a & 0xFF,
        )

    def composite_layer(
        self,
        dst: np.ndarray,
        src: np.ndarray,
        mask: Optional[np.ndarray] = None,
        clip_alpha: Optional[np.ndarray] = None,
        offset_x: int = 0,
        offset_y: int = 0,
        opacity: float = 1.0,
        blend_mode: int = BLEND_NORMAL,
        roi: Optional[tuple[int, int, int, int]] = None,
    ) -> None:
        dst_w, dst_h = _validate_rgba(dst, "dst")
        src_w, src_h = _validate_rgba(src, "src")
        mask_ptr = _opt_u8_ptr(mask, src_w, src_h, "mask")
        clip_ptr = _opt_u8_ptr(clip_alpha, dst_w, dst_h, "clip_alpha")
        rx, ry, rw, rh = roi if roi is not None else (0, 0, 0, 0)

        self.lib.composite_layer(
            dst.ctypes.data_as(U8_PTR), dst_w, dst_h,
            src.ctypes.data_as(U8_PTR), src_w, src_h,
            mask_ptr, clip_ptr,
            int(offset_x), int(offset_y),
            float(opacity), int(blend_mode),
            int(rx), int(ry), int(rw), int(rh),
        )

    def extract_alpha(
        self,
        src_rgba: np.ndarray,
        dst_alpha: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        w, h = _validate_rgba(src_rgba, "src_rgba")
        if dst_alpha is None:
            dst_alpha = np.empty((h, w), dtype=np.uint8)
        else:
            _validate_mask_2d(dst_alpha, expected_w=w, expected_h=h, name="dst_alpha")
        self.lib.extract_alpha(
            src_rgba.ctypes.data_as(U8_PTR),
            dst_alpha.ctypes.data_as(U8_PTR),
            w, h,
        )
        return dst_alpha

    def brush_dab(
        self,
        dst: np.ndarray,
        cx: float,
        cy: float,
        radius: float,
        hardness: float,
        color: tuple[int, int, int, int],
        opacity: float = 1.0,
        flow: float = 1.0,
        is_eraser: bool = False,
        selection_mask: Optional[np.ndarray] = None,
    ) -> Optional[tuple[int, int, int, int]]:
        w, h = _validate_rgba(dst, "dst")
        sel_ptr = _opt_u8_ptr(selection_mask, w, h, "selection_mask")
        dirty = np.zeros(4, dtype=np.int32)
        r, g, b, a = color
        self.lib.brush_dab_rgba(
            dst.ctypes.data_as(U8_PTR), w, h, sel_ptr,
            float(cx), float(cy), float(radius), float(hardness),
            r & 0xFF, g & 0xFF, b & 0xFF, a & 0xFF,
            float(opacity), float(flow), 1 if is_eraser else 0,
            dirty.ctypes.data_as(INT_PTR),
        )
        return _dirty_rect_to_tuple(dirty)

    def brush_stroke(
        self,
        dst: np.ndarray,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        radius: float,
        hardness: float,
        spacing: float,
        color: tuple[int, int, int, int],
        opacity: float = 1.0,
        flow: float = 1.0,
        is_eraser: bool = False,
        selection_mask: Optional[np.ndarray] = None,
    ) -> Optional[tuple[int, int, int, int]]:
        w, h = _validate_rgba(dst, "dst")
        sel_ptr = _opt_u8_ptr(selection_mask, w, h, "selection_mask")
        dirty = np.zeros(4, dtype=np.int32)
        r, g, b, a = color
        self.lib.brush_stroke_rgba(
            dst.ctypes.data_as(U8_PTR), w, h, sel_ptr,
            float(x0), float(y0), float(x1), float(y1),
            float(radius), float(hardness), float(spacing),
            r & 0xFF, g & 0xFF, b & 0xFF, a & 0xFF,
            float(opacity), float(flow), 1 if is_eraser else 0,
            dirty.ctypes.data_as(INT_PTR),
        )
        return _dirty_rect_to_tuple(dirty)

    def brush_stroke_mask(
        self,
        mask: np.ndarray,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        radius: float,
        hardness: float,
        spacing: float,
        value: int,
        opacity: float = 1.0,
        flow: float = 1.0,
        selection_mask: Optional[np.ndarray] = None,
    ) -> Optional[tuple[int, int, int, int]]:
        w, h = _validate_mask_2d(mask, name="mask")
        sel_ptr = _opt_u8_ptr(selection_mask, w, h, "selection_mask")
        dirty = np.zeros(4, dtype=np.int32)
        self.lib.brush_stroke_mask(
            mask.ctypes.data_as(U8_PTR), w, h, sel_ptr,
            float(x0), float(y0), float(x1), float(y1),
            float(radius), float(hardness), float(spacing),
            value & 0xFF, float(opacity), float(flow),
            dirty.ctypes.data_as(INT_PTR),
        )
        return _dirty_rect_to_tuple(dirty)

    def smudge_stroke(
        self,
        dst: np.ndarray,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        radius: float,
        hardness: float,
        strength: float,
        selection_mask: Optional[np.ndarray] = None,
    ) -> Optional[tuple[int, int, int, int]]:
        w, h = _validate_rgba(dst, "dst")
        sel_ptr = _opt_u8_ptr(selection_mask, w, h, "selection_mask")
        dirty = np.zeros(4, dtype=np.int32)
        self.lib.smudge_stroke_rgba(
            dst.ctypes.data_as(U8_PTR), w, h, sel_ptr,
            float(x0), float(y0), float(x1), float(y1),
            float(radius), float(hardness), float(strength),
            dirty.ctypes.data_as(INT_PTR),
        )
        return _dirty_rect_to_tuple(dirty)

    def blur_sharpen_dab(
        self,
        dst: np.ndarray,
        cx: float,
        cy: float,
        radius: float,
        hardness: float,
        strength: float,
        mode: int = FILTER_MODE_BLUR,
        kernel_size: int = 3,
        selection_mask: Optional[np.ndarray] = None,
    ) -> Optional[tuple[int, int, int, int]]:
        w, h = _validate_rgba(dst, "dst")
        sel_ptr = _opt_u8_ptr(selection_mask, w, h, "selection_mask")
        dirty = np.zeros(4, dtype=np.int32)
        self.lib.blur_sharpen_dab_rgba(
            dst.ctypes.data_as(U8_PTR), w, h, sel_ptr,
            float(cx), float(cy), float(radius), float(hardness),
            float(strength), int(mode), int(kernel_size),
            dirty.ctypes.data_as(INT_PTR),
        )
        return _dirty_rect_to_tuple(dirty)

    def gaussian_blur(
        self,
        src: np.ndarray,
        radius: int,
        dst: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        w, h = _validate_rgba(src, "src")
        if dst is None:
            dst = np.empty_like(src)
        else:
            dw, dh = _validate_rgba(dst, "dst")
            if (dw, dh) != (w, h):
                raise ValueError(f"dst shape mismatch: expected ({h}, {w}, 4), got ({dh}, {dw}, 4)")
        self.lib.gaussian_blur_rgba(
            src.ctypes.data_as(U8_PTR),
            dst.ctypes.data_as(U8_PTR),
            w, h, int(radius),
        )
        return dst

    def chroma_key(
        self,
        rgba: np.ndarray,
        key_rgb: tuple[int, int, int],
        tolerance: float = 40.0,
        softness: float = 20.0,
        despill: float = 0.5,
    ) -> None:
        w, h = _validate_rgba(rgba, "rgba")
        kr, kg, kb = key_rgb
        self.lib.chroma_key_rgba(
            rgba.ctypes.data_as(U8_PTR), w, h,
            kr & 0xFF, kg & 0xFF, kb & 0xFF,
            float(tolerance), float(softness), float(despill),
        )

    def flood_fill(
        self,
        rgba: np.ndarray,
        seed_x: int,
        seed_y: int,
        fill_rgba: tuple[int, int, int, int],
        tolerance: int = 32,
        contiguous: bool = True,
        selection_mask: Optional[np.ndarray] = None,
    ) -> tuple[int, Optional[tuple[int, int, int, int]]]:
        w, h = _validate_rgba(rgba, "rgba")
        sel_ptr = _opt_u8_ptr(selection_mask, w, h, "selection_mask")
        dirty = np.zeros(4, dtype=np.int32)
        fr, fg, fb, fa = fill_rgba
        filled = self.lib.flood_fill_rgba(
            rgba.ctypes.data_as(U8_PTR), w, h, sel_ptr,
            int(seed_x), int(seed_y),
            fr & 0xFF, fg & 0xFF, fb & 0xFF, fa & 0xFF,
            int(tolerance), 1 if contiguous else 0,
            dirty.ctypes.data_as(INT_PTR),
        )
        return int(filled), _dirty_rect_to_tuple(dirty)

    def magic_wand(
        self,
        rgba: np.ndarray,
        seed_x: int,
        seed_y: int,
        tolerance: int = 32,
        contiguous: bool = True,
        out_mask: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, int, Optional[tuple[int, int, int, int]]]:
        w, h = _validate_rgba(rgba, "rgba")
        if out_mask is None:
            out_mask = np.zeros((h, w), dtype=np.uint8)
        else:
            _validate_mask_2d(out_mask, expected_w=w, expected_h=h, name="out_mask")
        dirty = np.zeros(4, dtype=np.int32)
        count = self.lib.magic_wand_select(
            rgba.ctypes.data_as(U8_PTR), w, h,
            int(seed_x), int(seed_y),
            int(tolerance), 1 if contiguous else 0,
            out_mask.ctypes.data_as(U8_PTR),
            dirty.ctypes.data_as(INT_PTR),
        )
        return out_mask, int(count), _dirty_rect_to_tuple(dirty)

    def connected_components_alpha(
        self,
        rgba: np.ndarray,
        alpha_threshold: int = 0,
        min_area: int = 1,
        padding: int = 0,
        return_labels: bool = False,
    ) -> tuple[list[ComponentBox], Optional[np.ndarray]]:
        w, h = _validate_rgba(rgba, "rgba")
        labels_arr = np.zeros((h, w), dtype=np.int32) if return_labels else None
        labels_ptr = labels_arr.ctypes.data_as(I32_PTR) if labels_arr is not None else I32_PTR()

        # First determine exact count if large, or allocate 4096 boxes directly
        max_boxes = 4096
        box_array = (CComponentBox * max_boxes)()
        count = self.lib.connected_components_alpha(
            rgba.ctypes.data_as(U8_PTR), w, h,
            alpha_threshold & 0xFF, int(min_area), int(padding),
            labels_ptr, box_array, max_boxes,
        )
        if count == max_boxes:
            exact_total = self.lib.connected_components_alpha(
                rgba.ctypes.data_as(U8_PTR), w, h,
                alpha_threshold & 0xFF, int(min_area), int(padding),
                I32_PTR(), ctypes.POINTER(CComponentBox)(), 0,
            )
            if exact_total > max_boxes:
                box_array = (CComponentBox * exact_total)()
                count = self.lib.connected_components_alpha(
                    rgba.ctypes.data_as(U8_PTR), w, h,
                    alpha_threshold & 0xFF, int(min_area), int(padding),
                    labels_ptr, box_array, exact_total,
                )

        boxes = [
            ComponentBox(
                x=int(box_array[i].x),
                y=int(box_array[i].y),
                width=int(box_array[i].width),
                height=int(box_array[i].height),
                area=int(box_array[i].area),
                label=int(box_array[i].label),
            )
            for i in range(count)
        ]
        return boxes, labels_arr

    def find_alpha_bounds(
        self,
        rgba: np.ndarray,
        alpha_threshold: int = 0,
    ) -> Optional[tuple[int, int, int, int]]:
        w, h = _validate_rgba(rgba, "rgba")
        rect = np.zeros(4, dtype=np.int32)
        non_empty = self.lib.find_alpha_bounds(
            rgba.ctypes.data_as(U8_PTR), w, h,
            alpha_threshold & 0xFF,
            rect.ctypes.data_as(INT_PTR),
        )
        if not non_empty:
            return None
        return _dirty_rect_to_tuple(rect)

    def bilinear_resize(
        self,
        src: np.ndarray,
        dst_w: int,
        dst_h: int,
        dst: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        src_w, src_h = _validate_rgba(src, "src")
        if dst_w <= 0 or dst_h <= 0:
            raise ValueError(f"Destination size must be positive, got ({dst_w}, {dst_h})")
        if dst is None:
            dst = np.empty((dst_h, dst_w, 4), dtype=np.uint8)
        else:
            dw, dh = _validate_rgba(dst, "dst")
            if (dw, dh) != (dst_w, dst_h):
                raise ValueError(f"dst shape mismatch: expected ({dst_h}, {dst_w}, 4), got ({dh}, {dw}, 4)")
        self.lib.bilinear_resize_rgba(
            src.ctypes.data_as(U8_PTR), src_w, src_h,
            dst.ctypes.data_as(U8_PTR), int(dst_w), int(dst_h),
        )
        return dst


_ENGINE_INSTANCE: Optional[CoreEngineBridge] = None


def get_engine() -> CoreEngineBridge:
    """Return the singleton CoreEngineBridge instance, loading/compiling DLL on demand."""
    global _ENGINE_INSTANCE
    if _ENGINE_INSTANCE is None:
        _ENGINE_INSTANCE = CoreEngineBridge()
    return _ENGINE_INSTANCE
