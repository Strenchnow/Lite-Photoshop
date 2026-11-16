"""Multi-threaded batch folder image processing pipeline."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image

from app.core.c_bridge import get_engine


SUPPORTED_INPUT_EXTENSIONS = frozenset(
    {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tga", ".tif", ".tiff"}
)

_FORMAT_EXTENSIONS = {
    "PNG": ".png",
    "JPEG": ".jpg",
    "JPG": ".jpg",
    "WEBP": ".webp",
}


@dataclass
class BatchOperationConfig:
    """Configuration for BatchProcessor pipeline."""

    input_dir: Path
    output_dir: Path
    output_format: str = "PNG"
    quality: int = 90
    ai_remove_bg: bool = False
    chroma_key_enabled: bool = False
    chroma_key_rgb: tuple[int, int, int] = (0, 255, 0)
    chroma_tolerance: float = 40.0
    chroma_softness: float = 20.0
    chroma_despill: float = 0.5
    trim_transparent: bool = False
    resize_width: Optional[int] = None
    resize_height: Optional[int] = None
    max_workers: int = 4
    allow_ai_download: bool = True
    offline_ai_fallback: bool = False
    ai_model_path: Optional[Path] = None


@dataclass
class BatchResult:
    """Summary counts and output paths returned by BatchProcessor.run()."""

    total: int
    succeeded: int
    failed: int
    output_paths: list[Path] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)


def _normalize_format(fmt: str) -> tuple[str, str]:
    key = fmt.strip().upper()
    if key == "JPG":
        key = "JPEG"
    if key not in ("PNG", "JPEG", "WEBP"):
        raise ValueError(f"Unsupported batch output format: {fmt!r} (expected PNG, JPEG, or WEBP)")
    return key, _FORMAT_EXTENSIONS[key]


def _save_rgba_image(rgba: np.ndarray, out_path: Path, pil_format: str, quality: int) -> None:
    q = max(1, min(100, int(quality)))
    pil_rgba = Image.fromarray(rgba, mode="RGBA")
    if pil_format == "JPEG":
        rgb_bg = Image.new("RGB", pil_rgba.size, (255, 255, 255))
        rgb_bg.paste(pil_rgba, mask=pil_rgba.split()[3])
        rgb_bg.save(out_path, format="JPEG", quality=q, optimize=True)
    elif pil_format == "WEBP":
        pil_rgba.save(out_path, format="WEBP", quality=q, method=4)
    else:
        pil_rgba.save(out_path, format="PNG", compress_level=6)


class BatchProcessor:
    """Multi-threaded batch processor executing AI removal, chroma key, trim, resize, and save."""

    def __init__(self, config: BatchOperationConfig) -> None:
        self.config = config
        self._ai_lock = threading.Lock()

    def _collect_jobs(self, input_dir: Path, output_dir: Path, ext: str) -> list[tuple[Path, Path]]:
        files = sorted(
            p for p in input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_INPUT_EXTENSIONS
        )
        used_names: set[str] = set()
        jobs: list[tuple[Path, Path]] = []
        for src in files:
            candidate = f"{src.stem}{ext}"
            if candidate.lower() in used_names:
                suffix_idx = 1
                while f"{src.stem}_{suffix_idx}{ext}".lower() in used_names:
                    suffix_idx += 1
                candidate = f"{src.stem}_{suffix_idx}{ext}"
            used_names.add(candidate.lower())
            jobs.append((src, output_dir / candidate))
        return jobs

    def _process_one(self, src_path: Path, dst_path: Path, pil_format: str) -> Path:
        cfg = self.config
        with Image.open(src_path) as img:
            rgba = np.ascontiguousarray(np.asarray(img.convert("RGBA"), dtype=np.uint8))

        if cfg.ai_remove_bg:
            from app.core.ai_bg_remove import generate_alpha_matte

            with self._ai_lock:
                matte = generate_alpha_matte(
                    rgba,
                    model_path=cfg.ai_model_path,
                    allow_download=cfg.allow_ai_download,
                    offline_fallback=cfg.offline_ai_fallback,
                )
            combined = (rgba[:, :, 3].astype(np.uint16) * matte.astype(np.uint16) + 127) // 255
            rgba[:, :, 3] = combined.astype(np.uint8)

        engine = get_engine()

        if cfg.chroma_key_enabled:
            engine.chroma_key(
                rgba,
                key_rgb=cfg.chroma_key_rgb,
                tolerance=float(cfg.chroma_tolerance),
                softness=float(cfg.chroma_softness),
                despill=float(cfg.chroma_despill),
            )

        if cfg.trim_transparent:
            bounds = engine.find_alpha_bounds(rgba, alpha_threshold=0)
            if bounds is not None:
                bx, by, bw, bh = bounds
                rgba = np.ascontiguousarray(rgba[by : by + bh, bx : bx + bw].copy(), dtype=np.uint8)

        cur_h, cur_w = int(rgba.shape[0]), int(rgba.shape[1])
        target_w = cfg.resize_width if (cfg.resize_width is not None and cfg.resize_width > 0) else None
        target_h = cfg.resize_height if (cfg.resize_height is not None and cfg.resize_height > 0) else None

        if target_w is not None and target_h is None:
            target_h = max(1, int(round(cur_h * (target_w / max(cur_w, 1)))))
        elif target_h is not None and target_w is None:
            target_w = max(1, int(round(cur_w * (target_h / max(cur_h, 1)))))

        if target_w is not None and target_h is not None and (target_w != cur_w or target_h != cur_h):
            rgba = engine.bilinear_resize(rgba, int(target_w), int(target_h))

        _save_rgba_image(rgba, dst_path, pil_format=pil_format, quality=cfg.quality)
        return dst_path

    def run(
        self,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> BatchResult:
        cfg = self.config
        input_dir = Path(cfg.input_dir)
        output_dir = Path(cfg.output_dir)
        if not input_dir.is_dir():
            raise FileNotFoundError(f"Batch input directory does not exist: {input_dir}")

        pil_format, ext = _normalize_format(cfg.output_format)
        output_dir.mkdir(parents=True, exist_ok=True)
        jobs = self._collect_jobs(input_dir, output_dir, ext)
        total = len(jobs)
        if total == 0:
            return BatchResult(total=0, succeeded=0, failed=0, output_paths=[], errors={})

        # Ensure C engine singleton is initialized prior to spawning workers
        get_engine()

        # Pre-resolve/download ONNX model once before fanning out across threads
        if cfg.ai_remove_bg and not cfg.offline_ai_fallback:
            from app.core.ai_bg_remove import ensure_model_available

            ensure_model_available(model_path=cfg.ai_model_path, allow_download=cfg.allow_ai_download)

        workers = max(1, int(cfg.max_workers))
        output_paths: list[Path] = []
        errors: dict[str, str] = {}
        done = 0

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {}
            for src_path, dst_path in jobs:
                if cancel_event is not None and cancel_event.is_set():
                    break
                fut = executor.submit(self._process_one, src_path, dst_path, pil_format)
                future_map[fut] = src_path

            for fut in as_completed(future_map):
                src_path = future_map[fut]
                done += 1
                try:
                    out_file = fut.result()
                    output_paths.append(out_file)
                except Exception as exc:
                    errors[src_path.name] = str(exc)
                if progress_callback is not None:
                    progress_callback(done, total, src_path.name)

        output_paths.sort()
        return BatchResult(
            total=total,
            succeeded=len(output_paths),
            failed=len(errors),
            output_paths=output_paths,
            errors=errors,
        )
