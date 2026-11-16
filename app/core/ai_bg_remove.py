"""Lazy-loaded ONNX AI background removal (u2netp) with immediate session cleanup."""

from __future__ import annotations

import gc
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from app.core.document import CallableCommand, DocumentSession
from app.core.layer import LayerMask


U2NETP_MODEL_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx"
U2NETP_MODEL_FILENAME = "u2netp.onnx"
U2NETP_INPUT_SIZE = (320, 320)


def get_model_path(model_path: Optional[Path] = None) -> Path:
    """Resolve path to u2netp.onnx in workspace assets/models or PyInstaller _MEIPASS."""
    if model_path is not None:
        return Path(model_path)

    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        meipass_model = Path(sys._MEIPASS) / "assets" / "models" / U2NETP_MODEL_FILENAME  # type: ignore[attr-defined]
        if meipass_model.exists():
            return meipass_model
        exe_model = Path(sys.executable).resolve().parent / "assets" / "models" / U2NETP_MODEL_FILENAME
        if exe_model.exists():
            return exe_model
        return meipass_model

    project_root = Path(__file__).resolve().parents[2]
    return project_root / "assets" / "models" / U2NETP_MODEL_FILENAME


resolve_model_path = get_model_path


def download_u2netp_model(dest_path: Path, timeout: float = 60.0) -> Path:
    """Download u2netp.onnx atomically into `dest_path`."""
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(".onnx.tmp")

    req = urllib.request.Request(
        U2NETP_MODEL_URL,
        headers={"User-Agent": "LitePhotoEditor/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp, tmp_path.open("wb") as fh:
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                fh.write(chunk)
        if not tmp_path.exists() or tmp_path.stat().st_size == 0:
            raise RuntimeError(f"Downloaded empty file from {U2NETP_MODEL_URL}")
        tmp_path.replace(dest_path)
        return dest_path
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, RuntimeError) as exc:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise RuntimeError(f"Failed to download {U2NETP_MODEL_FILENAME} from {U2NETP_MODEL_URL}: {exc}") from exc


def ensure_model_available(
    model_path: Optional[Path] = None,
    allow_download: bool = True,
    timeout: float = 60.0,
) -> Path:
    """Ensure u2netp.onnx exists on disk, optionally downloading it on first use."""
    target = get_model_path(model_path)
    if target.is_file() and target.stat().st_size > 0:
        return target
    if not allow_download:
        raise FileNotFoundError(
            f"AI segmentation model not found at '{target}' and auto-download is disabled."
        )
    return download_u2netp_model(target, timeout=timeout)


def _validate_image_array(rgba: np.ndarray) -> tuple[int, int]:
    if not isinstance(rgba, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(rgba).__name__}")
    if rgba.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image buffer, got {rgba.dtype}")
    if rgba.ndim != 3 or rgba.shape[2] not in (3, 4):
        raise ValueError(f"Expected shape (H, W, 3|4), got {rgba.shape}")
    h, w = int(rgba.shape[0]), int(rgba.shape[1])
    if h <= 0 or w <= 0:
        raise ValueError(f"Image dimensions must be positive, got ({w}, {h})")
    return w, h


def _preprocess_for_u2net(rgba: np.ndarray, target_size: tuple[int, int] = U2NETP_INPUT_SIZE) -> np.ndarray:
    target_w, target_h = int(target_size[0]), int(target_size[1])
    rgb = np.ascontiguousarray(rgba[:, :, :3], dtype=np.uint8)
    resized = Image.fromarray(rgb, mode="RGB").resize((target_w, target_h), Image.Resampling.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    chw = np.transpose(arr, (2, 0, 1))
    return np.ascontiguousarray(chw[np.newaxis, :, :, :], dtype=np.float32)


def _postprocess_u2net_output(pred: np.ndarray, orig_w: int, orig_h: int) -> np.ndarray:
    arr = np.asarray(pred, dtype=np.float32)
    while arr.ndim > 2:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Unexpected ONNX output tensor shape: {pred.shape}")

    pmin = float(np.min(arr))
    pmax = float(np.max(arr))
    if pmax - pmin > 1e-8:
        norm = (arr - pmin) / (pmax - pmin)
    else:
        norm = np.clip(arr, 0.0, 1.0)

    matte = np.clip(np.round(norm * 255.0), 0, 255).astype(np.uint8)
    if (matte.shape[1], matte.shape[0]) != (orig_w, orig_h):
        matte_img = Image.fromarray(matte, mode="L").resize((orig_w, orig_h), Image.Resampling.BILINEAR)
        matte = np.asarray(matte_img, dtype=np.uint8)
    return np.ascontiguousarray(matte, dtype=np.uint8)


def _fallback_alpha_matte(rgba: np.ndarray) -> np.ndarray:
    """Offline heuristic foreground matte used only when offline_fallback=True in tests."""
    h, w = int(rgba.shape[0]), int(rgba.shape[1])
    rgb = rgba[:, :, :3].astype(np.float32)
    border_samples = np.concatenate(
        [rgb[0, :, :], rgb[h - 1, :, :], rgb[:, 0, :], rgb[:, w - 1, :]],
        axis=0,
    )
    bg_color = np.median(border_samples, axis=0)
    dist = np.linalg.norm(rgb - bg_color[None, None, :], axis=2)
    # Soft ramp between background similarity and foreground contrast
    matte = np.clip((dist - 18.0) / 36.0, 0.0, 1.0) * 255.0
    return np.ascontiguousarray(np.round(matte).astype(np.uint8), dtype=np.uint8)


def _run_u2net_session(rgba: np.ndarray, model_file: Path) -> np.ndarray:
    """Lazily import onnxruntime, execute inference, and release session RAM immediately."""
    import onnxruntime as ort  # Strict lazy import: never at module top level

    orig_w, orig_h = _validate_image_array(rgba)
    session = None
    try:
        session = ort.InferenceSession(str(model_file), providers=["CPUExecutionProvider"])
        inputs = session.get_inputs()
        input_name = inputs[0].name if inputs else "input.1"
        target_h, target_w = U2NETP_INPUT_SIZE
        if inputs and len(inputs[0].shape) == 4:
            sh, sw = inputs[0].shape[2], inputs[0].shape[3]
            if isinstance(sh, int) and sh > 0 and isinstance(sw, int) and sw > 0:
                target_h, target_w = sh, sw
        input_tensor = _preprocess_for_u2net(rgba, target_size=(target_w, target_h))
        outputs = session.run(None, {input_name: input_tensor})
        pred = np.asarray(outputs[0], dtype=np.float32)
    finally:
        if session is not None:
            del session
        gc.collect()

    return _postprocess_u2net_output(pred, orig_w=orig_w, orig_h=orig_h)


def generate_alpha_matte(
    rgba: np.ndarray,
    model_path: Optional[Path] = None,
    allow_download: bool = True,
    offline_fallback: bool = False,
) -> np.ndarray:
    """Generate a uint8 (H, W) foreground alpha matte using lazy-loaded u2netp.onnx."""
    _validate_image_array(rgba)
    try:
        model_file = ensure_model_available(model_path=model_path, allow_download=allow_download)
    except (FileNotFoundError, RuntimeError):
        if offline_fallback:
            return _fallback_alpha_matte(rgba)
        raise

    return _run_u2net_session(rgba, model_file)


def apply_ai_background_removal(
    doc: DocumentSession,
    layer_index: Optional[int] = None,
    allow_download: bool = True,
    model_path: Optional[Path] = None,
    offline_fallback: bool = False,
) -> LayerMask:
    """Run AI segmentation on target layer and attach matte as a non-destructive LayerMask."""
    if not doc.layers:
        raise ValueError("Document has no layers")
    idx = doc.active_layer_index if layer_index is None else int(layer_index)
    if not (0 <= idx < len(doc.layers)):
        raise IndexError(f"Layer index out of range: {idx}")

    layer = doc.layers[idx]
    matte = generate_alpha_matte(
        layer.pixels,
        model_path=model_path,
        allow_download=allow_download,
        offline_fallback=offline_fallback,
    )
    new_mask = LayerMask(layer.width, layer.height, buffer=matte, enabled=True)
    old_mask = layer.mask

    def do(session: DocumentSession) -> None:
        layer.mask = new_mask

    def undo(session: DocumentSession) -> None:
        layer.mask = old_mask

    do(doc)
    doc.undo_stack.push(
        CallableCommand(
            undo_fn=undo,
            redo_fn=do,
            description=f"AI Remove Background '{layer.name}'",
            estimated_bytes=int(matte.nbytes),
        )
    )
    doc.modified = True
    doc.invalidate_composite()
    return new_mask
