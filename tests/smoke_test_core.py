"""Smoke test for Stages 1 & 2: C core engine, c_bridge, Layer/LayerMask, and DocumentSession."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.c_bridge import (
    FILTER_MODE_BLUR,
    FILTER_MODE_SHARPEN,
    get_engine,
)
from app.core.document import DocumentTabManager, ViewportTransform
from app.core.layer import BlendMode, Layer


def test_c_engine_and_bridge() -> None:
    engine = get_engine()
    assert engine.dll_path.exists(), f"Missing DLL: {engine.dll_path}"
    for sym in engine.REQUIRED_SYMBOLS:
        assert hasattr(engine.lib, sym), f"Missing exported symbol: {sym}"

    # 1. Clear & extract_alpha
    buf = np.zeros((64, 64, 4), dtype=np.uint8)
    engine.clear_buffer(buf, 100, 150, 200, 255)
    assert np.all(buf[0, 0] == [100, 150, 200, 255])
    alpha = engine.extract_alpha(buf)
    assert alpha.shape == (64, 64) and np.all(alpha == 255)

    # 2. Composite blend modes
    base = np.zeros((16, 16, 4), dtype=np.uint8)
    top = np.zeros((16, 16, 4), dtype=np.uint8)
    for mode, expected_rgb in (
        (BlendMode.NORMAL, (100, 150, 200)),
        (BlendMode.MULTIPLY, (78, 118, 157)),
        (BlendMode.SCREEN, (222, 232, 243)),
        (BlendMode.OVERLAY, (188, 210, 231)),
        (BlendMode.ADD, (255, 255, 255)),
    ):
        engine.clear_buffer(base, 200, 200, 200, 255)
        engine.clear_buffer(top, 100, 150, 200, 255)
        engine.composite_layer(base, top, blend_mode=int(mode))
        r, g, b, a = base[8, 8]
        assert a == 255
        assert abs(int(r) - expected_rgb[0]) <= 2, f"{mode}: got R={r}, expected {expected_rgb[0]}"
        assert abs(int(g) - expected_rgb[1]) <= 2, f"{mode}: got G={g}, expected {expected_rgb[1]}"
        assert abs(int(b) - expected_rgb[2]) <= 2, f"{mode}: got B={b}, expected {expected_rgb[2]}"

    # Also test Overlay dark branch (d < 0.5)
    engine.clear_buffer(base, 51, 51, 51, 255)  # 0.2
    engine.clear_buffer(top, 128, 128, 128, 255)  # ~0.5
    engine.composite_layer(base, top, blend_mode=int(BlendMode.OVERLAY))
    assert abs(int(base[8, 8, 0]) - 51) <= 2

    # 3. Brush stroke continuity + eraser + mask brush
    canvas = np.zeros((64, 64, 4), dtype=np.uint8)
    dirty = engine.brush_stroke(
        canvas, x0=8.0, y0=32.0, x1=56.0, y1=32.0,
        radius=4.0, hardness=1.0, spacing=0.25,
        color=(255, 0, 0, 255), opacity=1.0, flow=1.0,
    )
    assert dirty is not None
    # Every pixel along the horizontal stroke line must be filled continuously
    assert np.all(canvas[32, 8:57, 3] == 255), "Brush stroke must be continuous without gaps"

    engine.brush_dab(
        canvas, cx=32.0, cy=32.0, radius=3.0, hardness=1.0,
        color=(0, 0, 0, 255), opacity=1.0, flow=1.0, is_eraser=True,
    )
    assert canvas[32, 32, 3] == 0, "Eraser dab must clear alpha"

    mask = np.full((64, 64), 255, dtype=np.uint8)
    m_dirty = engine.brush_stroke_mask(
        mask, x0=10.0, y0=10.0, x1=20.0, y1=10.0,
        radius=2.0, hardness=1.0, spacing=0.25, value=0,
    )
    assert m_dirty is not None and np.all(mask[10, 10:21] == 0)

    # 4. Smudge + Local Blur / Sharpen + Global Gaussian Blur
    smudge_buf = np.zeros((32, 32, 4), dtype=np.uint8)
    smudge_buf[:, :12] = [255, 0, 0, 255]
    s_dirty = engine.smudge_stroke(
        smudge_buf, x0=8.0, y0=16.0, x1=20.0, y1=16.0,
        radius=4.0, hardness=0.8, strength=0.9,
    )
    assert s_dirty is not None and smudge_buf[16, 16, 3] > 0, "Smudge must carry color into transparent area"

    sharp_buf = smudge_buf.copy()
    engine.blur_sharpen_dab(sharp_buf, cx=12.0, cy=16.0, radius=5.0, hardness=0.5, strength=0.8, mode=FILTER_MODE_BLUR, kernel_size=3)
    engine.blur_sharpen_dab(sharp_buf, cx=12.0, cy=16.0, radius=5.0, hardness=0.5, strength=0.5, mode=FILTER_MODE_SHARPEN, kernel_size=5)
    blurred = engine.gaussian_blur(smudge_buf, radius=2)
    assert blurred.shape == smudge_buf.shape

    # 5. Chroma Key
    gs = np.zeros((32, 32, 4), dtype=np.uint8)
    gs[:, :] = [0, 255, 0, 255]
    gs[10:22, 10:22] = [220, 20, 20, 255]
    engine.chroma_key(gs, key_rgb=(0, 255, 0), tolerance=40.0, softness=10.0, despill=0.5)
    assert gs[0, 0, 3] == 0, "Green screen background must become transparent"
    assert gs[16, 16, 3] == 255, "Red foreground must remain opaque"

    # 6. Flood Fill & Magic Wand
    ff = np.zeros((32, 32, 4), dtype=np.uint8)
    ff[:, :] = [10, 10, 10, 255]
    ff[8:24, 8] = [255, 255, 255, 255]
    ff[8:24, 24] = [255, 255, 255, 255]
    ff[8, 8:25] = [255, 255, 255, 255]
    ff[24, 8:25] = [255, 255, 255, 255]
    filled_px, ff_dirty = engine.flood_fill(ff, seed_x=16, seed_y=16, fill_rgba=(0, 128, 255, 255), tolerance=5, contiguous=True)
    assert filled_px == 15 * 15 and ff_dirty == (9, 9, 15, 15)
    assert tuple(ff[16, 16]) == (0, 128, 255, 255)
    assert tuple(ff[0, 0]) == (10, 10, 10, 255)

    wand_mask, wand_cnt, _ = engine.magic_wand(ff, seed_x=16, seed_y=16, tolerance=5, contiguous=True)
    assert wand_cnt == 15 * 15 and wand_mask[16, 16] == 255 and wand_mask[0, 0] == 0

    # 7. Connected Components Labeling & Alpha Bounds & Resize
    sprites = np.zeros((64, 64, 4), dtype=np.uint8)
    sprites[4:14, 4:14, 3] = 255   # 10x10 = 100 px
    sprites[30:45, 35:50, 3] = 255 # 15x15 = 225 px
    sprites[60:62, 60:62, 3] = 255 # 2x2 = 4 px noise
    boxes, labels = engine.connected_components_alpha(
        sprites, alpha_threshold=10, min_area=20, padding=2, return_labels=True,
    )
    assert len(boxes) == 2, f"Expected 2 sprite boxes, got {len(boxes)}"
    assert boxes[0].area == 100 and (boxes[0].x, boxes[0].y, boxes[0].width, boxes[0].height) == (2, 2, 14, 14)
    assert boxes[1].area == 225 and (boxes[1].x, boxes[1].y, boxes[1].width, boxes[1].height) == (33, 28, 19, 19)
    assert labels is not None and labels[60, 60] == 0

    bounds = engine.find_alpha_bounds(sprites, alpha_threshold=10)
    assert bounds == (4, 4, 58, 58)

    resized = engine.bilinear_resize(sprites, 32, 32)
    assert resized.shape == (32, 32, 4)


def test_document_delta_undo_and_tabs() -> None:
    engine = get_engine()
    mgr = DocumentTabManager()

    # Create two tabs to verify multi-tab isolation & zero-overhead inactive cache
    doc1 = mgr.new_document(512, 512, title="CanvasOne", initial_fill=(255, 255, 255, 255))
    comp1 = doc1.composite()
    assert comp1.shape == (512, 512, 4) and doc1._composited_cache is not None

    doc2 = mgr.new_document(256, 256, title="CanvasTwo", initial_fill=(0, 0, 0, 255))
    assert doc1.is_active_tab is False and doc1._composited_cache is None, "Inactive tab must release cache"
    assert doc2.is_active_tab is True and doc2.display_title == "CanvasTwo"

    # Switch back to doc1 and test Delta Undo/Redo
    mgr.set_active_tab(0)
    assert doc2._composited_cache is None
    assert doc1.is_active_tab is True

    layer = doc1.add_layer(name="PaintLayer")
    assert doc1.display_title == "CanvasOne *"

    with doc1.edit_layer_context(layer=layer, roi=(90, 90, 40, 40), description="Brush Dab"):
        engine.brush_dab(
            layer.pixels, cx=110.0, cy=110.0, radius=5.0, hardness=1.0,
            color=(255, 0, 0, 255),
        )

    last_cmd = doc1.undo_stack._undo_list[-1]
    assert last_cmd.memory_bytes < 4096, f"Delta undo should store tiny patch, stored {last_cmd.memory_bytes} bytes"
    assert tuple(layer.pixels[110, 110]) == (255, 0, 0, 255)

    doc1.undo()
    assert tuple(layer.pixels[110, 110]) == (0, 0, 0, 0), "Undo must restore transparent pixel"
    doc1.redo()
    assert tuple(layer.pixels[110, 110]) == (255, 0, 0, 255), "Redo must re-apply red dab"

    # Test LayerMask + Clipping Mask compositing
    base = doc1.add_layer(name="ClipBase")
    base.pixels[10:30, 10:30] = [0, 0, 255, 255]
    clipped = doc1.add_layer(name="ClipTop")
    clipped.clipping_mask = True
    clipped.pixels[0:40, 0:40] = [255, 255, 0, 255]
    clipped.create_mask(fill_value=255)
    clipped.mask.buffer[10:20, 10:20] = 0

    comp = doc1.composite()
    assert tuple(comp[5, 5]) == (255, 255, 255, 255), "Clipped layer outside base alpha must not draw"
    assert tuple(comp[15, 15]) == (0, 0, 255, 255), "LayerMask=0 must reveal ClipBase"
    assert tuple(comp[25, 25]) == (255, 255, 0, 255), "Clipped layer inside base alpha must draw"

    # Test ViewportTransform zoom-to-cursor invariance
    vp = ViewportTransform(zoom=1.0, pan_x=50.0, pan_y=30.0)
    cx_before, cy_before = vp.screen_to_canvas(200.0, 150.0)
    vp.zoom_at_cursor(200.0, 150.0, 2.5)
    cx_after, cy_after = vp.screen_to_canvas(200.0, 150.0)
    assert abs(cx_before - cx_after) < 1e-6 and abs(cy_before - cy_after) < 1e-6

    # Test 35 consecutive delta undo/redo steps, merge_down, center, and trim
    doc3 = mgr.new_document(200, 200, title="OpsTest")
    paint = doc3.active_layer
    for step in range(35):
        x = (step * 4) % 180 + 10
        with doc3.edit_layer_context(layer=paint, roi=(x - 2, 10, 5, 5), description=f"Dot {step}"):
            paint.pixels[12, x] = [step * 5 % 255, 100, 200, 255]
    assert doc3.undo_stack.undo_count == 35
    assert doc3.undo_stack.total_memory_bytes < 4096
    for _ in range(35):
        assert doc3.undo() is not None
    assert np.all(paint.pixels[:, :, 3] == 0)
    for _ in range(35):
        assert doc3.redo() is not None

    # Center & Trim transparent
    trim_doc = mgr.new_document(100, 100, title="TrimTest")
    trim_doc.active_layer.pixels[5:15, 5:15] = [10, 200, 10, 255]
    trim_doc.center_active_layer(horizontal=True, vertical=True, use_content_bounds=True)
    bounds = trim_doc.active_layer.get_non_transparent_bounds()
    assert bounds == (45, 45, 10, 10), f"Expected centered bounds (45, 45, 10, 10), got {bounds}"
    assert trim_doc.trim_transparent() is True
    assert (trim_doc.width, trim_doc.height) == (10, 10)
    trim_doc.undo()
    assert (trim_doc.width, trim_doc.height) == (100, 100)

    # Merge down
    merge_doc = mgr.new_document(32, 32, title="MergeTest", initial_fill=(100, 100, 100, 255))
    top_l = merge_doc.add_layer(name="Top", fill_rgba=(255, 0, 0, 255))
    top_l.opacity = 0.5
    merged = merge_doc.merge_down()
    assert len(merge_doc.layers) == 1 and merged.pixels[0, 0, 0] > 150
    merge_doc.undo()
    assert len(merge_doc.layers) == 2


if __name__ == "__main__":
    test_c_engine_and_bridge()
    test_document_delta_undo_and_tabs()
    print("ALL SMOKE TESTS PASSED")
