"""Comprehensive headless integration smoke test for LitePhotoEditor."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_imports_and_lazy_loading() -> None:
    # Import all core, tool, UI, entry-point, and build modules
    import app  # noqa: F401
    import app.config.i18n  # noqa: F401
    import app.core  # noqa: F401
    import app.core.ai_bg_remove  # noqa: F401
    import app.core.batch  # noqa: F401
    import app.core.c_bridge  # noqa: F401
    import app.core.document  # noqa: F401
    import app.core.layer  # noqa: F401
    import app.core.splitter  # noqa: F401
    import app.tools  # noqa: F401
    import app.tools.base_tool  # noqa: F401
    import app.tools.brush_tools  # noqa: F401
    import app.tools.nav_tools  # noqa: F401
    import app.tools.select_tools  # noqa: F401
    import app.tools.utility_tools  # noqa: F401
    import app.ui  # noqa: F401
    import app.ui.canvas_view  # noqa: F401
    import app.ui.dialogs  # noqa: F401
    import app.ui.main_window  # noqa: F401
    import app.ui.panels  # noqa: F401
    import app.main  # noqa: F401
    import build_exe  # noqa: F401

    # Strict lazy loading: onnxruntime must NOT be loaded on module import
    assert "onnxruntime" not in sys.modules, (
        "onnxruntime was imported eagerly during module imports!"
    )


def test_all_15_tools() -> None:
    import numpy as np
    from PySide6.QtCore import Qt

    from app.core.document import DocumentSession
    from app.tools import (
        MOD_ALT,
        MOD_NONE,
        MOD_SHIFT,
        BlurSharpenTool,
        BrushTool,
        BucketFillTool,
        CropTool,
        EraserTool,
        EyedropperTool,
        FreeTransformTool,
        GradientTool,
        HandTool,
        LassoTool,
        MagicWandTool,
        MoveTool,
        RectMarqueeTool,
        SmudgeTool,
        TextTool,
        ToolContext,
        ToolType,
        center_both,
        center_horizontal,
        center_vertical,
        create_default_tools,
        extract_modifiers,
    )

    assert extract_modifiers(Qt.ShiftModifier | Qt.ControlModifier) == (1 | 2)

    picked_colors: list[tuple[int, int, int, int]] = []
    invalidations = [0]
    resizes = [0]
    ctx = ToolContext(
        on_color_picked=picked_colors.append,
        on_canvas_invalidated=lambda: invalidations.__setitem__(0, invalidations[0] + 1),
        on_document_resized=lambda: resizes.__setitem__(0, resizes[0] + 1),
    )
    tools = create_default_tools(ctx)
    assert len(tools) == 15
    expected_ids = {t.value for t in ToolType}
    assert set(tools.keys()) == expected_ids

    # 1. HandTool
    doc = DocumentSession(200, 200, title="ToolsDoc")
    hand = tools[ToolType.HAND.value]
    assert isinstance(hand, HandTool)
    hand.mouse_press(doc, 0, 0, button=1, sx=10.0, sy=20.0)
    hand.mouse_move(doc, 0, 0, buttons=1, sx=35.0, sy=50.0)
    hand.mouse_release(doc, 0, 0, button=1, sx=35.0, sy=50.0)
    assert (doc.viewport.pan_x, doc.viewport.pan_y) == (25.0, 30.0)
    doc.viewport.pan_x = 0.0
    doc.viewport.pan_y = 0.0

    # 2. MoveTool magnetic snapping + MOD_SHIFT override + center helpers
    move = tools[ToolType.MOVE.value]
    assert isinstance(move, MoveTool)
    doc.active_layer.pixels[50:70, 50:70] = [200, 100, 50, 255]  # 20x20 content at (50,50)
    # Snap left edge to canvas x=0 (requires offset_x == -50; try raw_ox = -47 within 6px)
    move.mouse_press(doc, 100.0, 100.0, button=1)
    move.mouse_move(doc, 53.0, 52.0, buttons=1, modifiers=MOD_NONE)
    assert doc.active_layer.offset_x == -50, f"Expected snap to left edge (-50), got {doc.active_layer.offset_x}"
    assert doc.active_layer.offset_y == -50, f"Expected snap to top edge (-50), got {doc.active_layer.offset_y}"
    assert ("vertical", 0.0) in ctx.snap_guides
    assert ("horizontal", 0.0) in ctx.snap_guides

    # Snap center to canvas center (hcenter = ox + 50 + 10 == 100 -> ox == 40)
    move.mouse_move(doc, 138.0, 138.0, buttons=1, modifiers=MOD_NONE)
    assert doc.active_layer.offset_x == 40
    assert ("vertical", 100.0) in ctx.snap_guides
    assert ("horizontal", 100.0) in ctx.snap_guides

    # MOD_SHIFT must bypass snapping and clear snap_guides
    move.mouse_move(doc, 138.0, 138.0, buttons=1, modifiers=MOD_SHIFT)
    assert doc.active_layer.offset_x == 38
    assert doc.active_layer.offset_y == 38
    assert ctx.snap_guides == []
    move.mouse_release(doc, 138.0, 138.0, button=1, modifiers=MOD_SHIFT)
    assert (doc.active_layer.offset_x, doc.active_layer.offset_y) == (38, 38)
    doc.undo()
    assert (doc.active_layer.offset_x, doc.active_layer.offset_y) == (0, 0)
    doc.redo()
    assert (doc.active_layer.offset_x, doc.active_layer.offset_y) == (38, 38)

    center_horizontal(doc)
    assert doc.active_layer.get_non_transparent_bounds()[0] == 90
    center_vertical(doc)
    assert doc.active_layer.get_non_transparent_bounds()[1] == 90
    doc.move_layer_offset(0, 0, 0)
    center_both(doc)
    assert doc.active_layer.get_non_transparent_bounds() == (90, 90, 20, 20)
    doc.move_layer_offset(0, 0, 0)

    # 3. BrushTool (normal stroke, Alt-eyedropper, offset layer, mask painting, locked guard)
    brush = tools[ToolType.BRUSH.value]
    assert isinstance(brush, BrushTool)
    paint_layer = doc.add_layer(name="Paint")
    paint_layer.offset_x = 10
    paint_layer.offset_y = 10
    ctx.fg_color = (255, 0, 0, 255)
    ctx.brush_radius = 4.0
    ctx.brush_hardness = 1.0
    ctx.brush_opacity = 1.0
    ctx.brush_flow = 1.0
    brush.mouse_press(doc, 30.0, 30.0, button=1)
    brush.mouse_move(doc, 60.0, 30.0, buttons=1)
    brush.mouse_release(doc, 60.0, 30.0, button=1)
    # Local x = [20..50], y = 20
    assert np.all(paint_layer.pixels[20, 20:51, 3] == 255)
    assert tuple(paint_layer.pixels[20, 35]) == (255, 0, 0, 255)
    doc.undo()
    assert paint_layer.pixels[20, 35, 3] == 0
    doc.redo()
    assert paint_layer.pixels[20, 35, 3] == 255

    # Alt-click eyedropper via BrushTool
    ctx.fg_color = (0, 0, 0, 255)
    brush.mouse_press(doc, 45.0, 30.0, button=1, modifiers=MOD_ALT)
    assert ctx.fg_color == (255, 0, 0, 255)
    assert picked_colors[-1] == (255, 0, 0, 255)

    # Mask painting via BrushTool
    paint_layer.create_mask(fill_value=255)
    ctx.edit_mask = True
    ctx.fg_color = (0, 0, 0, 255)
    brush.mouse_press(doc, 45.0, 30.0, button=1)
    brush.mouse_release(doc, 45.0, 30.0, button=1)
    assert paint_layer.mask.buffer[20, 35] == 0
    doc.undo()
    assert paint_layer.mask.buffer[20, 35] == 255
    ctx.edit_mask = False

    # Locked layer guard
    paint_layer.locked = True
    before_locked = paint_layer.pixels.copy()
    brush.mouse_press(doc, 80.0, 80.0, button=1)
    brush.mouse_release(doc, 80.0, 80.0, button=1)
    assert np.array_equal(paint_layer.pixels, before_locked)
    paint_layer.locked = False

    # 4. EraserTool
    eraser = tools[ToolType.ERASER.value]
    assert isinstance(eraser, EraserTool)
    eraser.mouse_press(doc, 45.0, 30.0, button=1)
    eraser.mouse_release(doc, 45.0, 30.0, button=1)
    assert paint_layer.pixels[20, 35, 3] == 0
    doc.undo()
    assert paint_layer.pixels[20, 35, 3] == 255

    # 5. SmudgeTool
    smudge = tools[ToolType.SMUDGE.value]
    assert isinstance(smudge, SmudgeTool)
    smudge_doc = DocumentSession(64, 64, title="SmudgeDoc")
    smudge_doc.active_layer.pixels[:, :16] = [0, 255, 0, 255]
    ctx.brush_radius = 5.0
    ctx.smudge_strength = 0.9
    smudge.mouse_press(smudge_doc, 10.0, 32.0, button=1)
    smudge.mouse_move(smudge_doc, 24.0, 32.0, buttons=1)
    smudge.mouse_release(smudge_doc, 24.0, 32.0, button=1)
    assert smudge_doc.active_layer.pixels[32, 20, 3] > 0
    smudge_doc.undo()
    assert smudge_doc.active_layer.pixels[32, 20, 3] == 0

    # 6. BlurSharpenTool
    blur_tool = tools[ToolType.BLUR_SHARPEN.value]
    assert isinstance(blur_tool, BlurSharpenTool)
    blur_doc = DocumentSession(64, 64, title="FilterDoc")
    blur_doc.active_layer.pixels[:, :32] = [255, 255, 255, 255]
    blur_doc.active_layer.pixels[:, 32:] = [0, 0, 0, 255]
    ctx.brush_radius = 6.0
    ctx.filter_mode = 0
    ctx.filter_kernel_size = 3
    ctx.filter_strength = 0.9
    blur_tool.mouse_press(blur_doc, 32.0, 32.0, button=1)
    blur_tool.mouse_move(blur_doc, 32.0, 40.0, buttons=1)
    blur_tool.mouse_release(blur_doc, 32.0, 40.0, button=1)
    assert blur_doc.active_layer.pixels[32, 32, 0] > 0
    ctx.filter_mode = 1
    ctx.filter_kernel_size = 5
    blur_tool.mouse_press(blur_doc, 32.0, 20.0, button=1)
    blur_tool.mouse_release(blur_doc, 32.0, 20.0, button=1)
    blur_doc.undo()

    # 7. RectMarqueeTool
    marquee = tools[ToolType.MARQUEE.value]
    assert isinstance(marquee, RectMarqueeTool)
    sel_doc = DocumentSession(100, 100, title="SelDoc")
    marquee.mouse_press(sel_doc, 10.0, 10.0, button=1)
    marquee.mouse_move(sel_doc, 30.0, 30.0, buttons=1)
    marquee.mouse_release(sel_doc, 30.0, 30.0, button=1)
    assert sel_doc.selection_mask is not None
    assert sel_doc.selection_mask[20, 20] == 255 and sel_doc.selection_mask[5, 5] == 0
    # Shift add
    marquee.mouse_press(sel_doc, 40.0, 40.0, button=1, modifiers=MOD_SHIFT)
    marquee.mouse_release(sel_doc, 50.0, 50.0, button=1, modifiers=MOD_SHIFT)
    assert sel_doc.selection_mask[20, 20] == 255 and sel_doc.selection_mask[45, 45] == 255
    # Alt subtract
    marquee.mouse_press(sel_doc, 15.0, 15.0, button=1, modifiers=MOD_ALT)
    marquee.mouse_release(sel_doc, 25.0, 25.0, button=1, modifiers=MOD_ALT)
    assert sel_doc.selection_mask[20, 20] == 0 and sel_doc.selection_mask[11, 11] == 255
    # Click clears
    marquee.mouse_press(sel_doc, 5.0, 5.0, button=1)
    marquee.mouse_release(sel_doc, 5.0, 5.0, button=1)
    assert sel_doc.selection_mask is None

    # 8. LassoTool (live draw_overlay, rasterization, Shift-add, Alt-subtract, Esc deselect)
    from PySide6.QtGui import QImage as _QImg, QPainter as _QPainter

    lasso = tools[ToolType.LASSO.value]
    assert isinstance(lasso, LassoTool)
    lasso.mouse_press(sel_doc, 10.0, 10.0, button=1)
    lasso.mouse_move(sel_doc, 50.0, 10.0, buttons=1)
    lasso.mouse_move(sel_doc, 30.0, 50.0, buttons=1)
    overlay_img = _QImg(100, 100, _QImg.Format.Format_ARGB32)
    overlay_painter = _QPainter(overlay_img)
    lasso.draw_overlay(overlay_painter, sel_doc, sel_doc.viewport)
    overlay_painter.end()
    lasso.mouse_release(sel_doc, 10.0, 10.0, button=1)
    assert sel_doc.selection_mask is not None
    assert sel_doc.selection_mask[25, 30] == 255
    assert sel_doc.selection_mask[5, 5] == 0
    # Shift add second triangle
    lasso.mouse_press(sel_doc, 60.0, 60.0, button=1, modifiers=MOD_SHIFT)
    lasso.mouse_move(sel_doc, 90.0, 60.0, buttons=1, modifiers=MOD_SHIFT)
    lasso.mouse_move(sel_doc, 75.0, 90.0, buttons=1, modifiers=MOD_SHIFT)
    lasso.mouse_release(sel_doc, 60.0, 60.0, button=1, modifiers=MOD_SHIFT)
    assert sel_doc.selection_mask[25, 30] == 255 and sel_doc.selection_mask[70, 75] == 255
    # Esc clears selection
    assert lasso.key_press(sel_doc, Qt.Key_Escape) is True
    assert sel_doc.selection_mask is None

    # 9. MagicWandTool (active layer + multi-layer transparent active layer fallback)
    wand = tools[ToolType.MAGIC_WAND.value]
    assert isinstance(wand, MagicWandTool)
    sel_doc.active_layer.pixels[20:40, 20:40] = [10, 200, 220, 255]
    ctx.wand_tolerance = 10
    ctx.wand_contiguous = True
    wand.mouse_press(sel_doc, 25.0, 25.0, button=1)
    assert sel_doc.selection_mask is not None
    assert int(np.sum(sel_doc.selection_mask == 255)) == 400
    assert wand.key_press(sel_doc, Qt.Key_Escape) is True
    assert sel_doc.selection_mask is None

    # Transparent top active layer clicking over visible bottom layer must still select visible artwork
    top_empty = sel_doc.add_layer(name="EmptyTop")
    assert sel_doc.active_layer is top_empty
    wand.mouse_press(sel_doc, 25.0, 25.0, button=1)
    assert sel_doc.selection_mask is not None
    assert int(np.sum(sel_doc.selection_mask == 255)) == 400
    sel_doc.clear_selection()

    # 10. CropTool (Esc cancel, Enter apply + undo, aspect ratio presets, swap, Shift free override)
    crop = tools[ToolType.CROP.value]
    assert isinstance(crop, CropTool)
    crop_doc = DocumentSession(100, 100, title="CropDoc")
    crop_doc.active_layer.pixels[20:40, 20:40] = [255, 0, 255, 255]
    ctx.set_crop_preset("free")
    crop.mouse_press(crop_doc, 15.0, 15.0, button=1)
    crop.mouse_move(crop_doc, 55.0, 45.0, buttons=1)
    crop.mouse_release(crop_doc, 55.0, 45.0, button=1)
    assert crop.crop_rect == (15, 15, 40, 30)
    assert crop.key_press(crop_doc, Qt.Key_Escape) is True
    assert crop.crop_rect is None and (crop_doc.width, crop_doc.height) == (100, 100)

    # Preset 16:9 + Swap -> 9:16 + Shift free override
    ctx.set_crop_preset("16:9")
    crop.mouse_press(crop_doc, 10.0, 10.0, button=1)
    crop.mouse_move(crop_doc, 90.0, 90.0, buttons=1)
    crop.mouse_release(crop_doc, 90.0, 90.0, button=1)
    assert crop.crop_rect is not None
    _, _, cw16, ch16 = crop.crop_rect
    assert abs((cw16 / ch16) - (16.0 / 9.0)) < 0.08
    assert ctx.swap_crop_orientation() == "9:16"
    crop.apply_aspect_ratio_to_current_rect(crop_doc)
    _, _, cw9, ch9 = crop.crop_rect
    assert abs((cw9 / ch9) - (9.0 / 16.0)) < 0.08
    # Holding Shift temporarily overrides locked ratio
    crop.mouse_press(crop_doc, 10.0, 10.0, button=1, modifiers=MOD_SHIFT)
    crop.mouse_move(crop_doc, 60.0, 30.0, buttons=1, modifiers=MOD_SHIFT)
    crop.mouse_release(crop_doc, 60.0, 30.0, button=1, modifiers=MOD_SHIFT)
    assert crop.crop_rect == (10, 10, 50, 20)
    ctx.set_crop_preset("free")
    crop.cancel_crop()

    crop.mouse_press(crop_doc, 15.0, 15.0, button=1)
    crop.mouse_release(crop_doc, 55.0, 45.0, button=1)
    assert crop.key_press(crop_doc, Qt.Key_Return) is True
    assert (crop_doc.width, crop_doc.height) == (40, 30)
    assert (crop_doc.active_layer.offset_x, crop_doc.active_layer.offset_y) == (-15, -15)
    crop_doc.undo()
    assert (crop_doc.width, crop_doc.height) == (100, 100)

    # 11. FreeTransformTool (non-ghosting suppressed layer, magnetic snapping, aspect-lock Shift, rotation, Enter/Esc)
    ft = tools[ToolType.TRANSFORM.value]
    assert isinstance(ft, FreeTransformTool)
    ft_doc = DocumentSession(120, 120, title="TransformDoc")
    ft_doc.active_layer.pixels[20:40, 20:40] = [0, 128, 255, 255]
    ft.on_activate(ft_doc)
    assert ft.active is True
    assert ft_doc.suppressed_layer_id == ft_doc.active_layer.id
    assert ft_doc.composite()[25, 25, 3] == 0, "Original layer must be suppressed during active transform"
    assert (ft.orig_w, ft.orig_h) == (20.0, 20.0)

    # Magnetic snapping when moving transform box near left/top edge (center 13 -> left=3 -> snaps to 0, center=10)
    ft.mouse_press(ft_doc, 30.0, 30.0, button=1)
    ft.mouse_move(ft_doc, 13.0, 13.0, buttons=1, modifiers=MOD_NONE)
    assert abs(ft.center_x - 10.0) < 1e-4 and abs(ft.center_y - 10.0) < 1e-4
    assert ("vertical", 0.0) in ctx.snap_guides and ("horizontal", 0.0) in ctx.snap_guides
    ft.mouse_move(ft_doc, 13.0, 13.0, buttons=1, modifiers=MOD_SHIFT)
    assert ctx.snap_guides == []
    ft.mouse_release(ft_doc, 30.0, 30.0, button=1)
    ft.center_x, ft.center_y = 30.0, 30.0

    # Drag SE corner handle (40, 40) to (70, 55) with MOD_SHIFT -> aspect ratio stays 1:1
    ft.mouse_press(ft_doc, 40.0, 40.0, button=1)
    ft.mouse_move(ft_doc, 70.0, 55.0, buttons=1, modifiers=MOD_SHIFT)
    ft.mouse_release(ft_doc, 70.0, 55.0, button=1, modifiers=MOD_SHIFT)
    assert abs(ft.box_w - ft.box_h) < 1e-4 and ft.box_w > 30.0
    assert ft.key_press(ft_doc, Qt.Key_Escape) is True
    assert ft.active is False
    assert ft_doc.suppressed_layer_id is None
    assert ft_doc.composite()[25, 25, 3] == 255, "Cancelled transform must restore original layer rendering"

    ft.begin_transform(ft_doc)
    ft.set_transform(scale_x=2.0, scale_y=2.0, angle_deg=15.0)
    assert ft.key_press(ft_doc, Qt.Key_Return) is True
    bounds_after = ft_doc.active_layer.get_non_transparent_bounds()
    assert bounds_after is not None and bounds_after[2] > 35 and bounds_after[3] > 35
    ft_doc.undo()
    assert ft_doc.active_layer.get_non_transparent_bounds() == (20, 20, 20, 20)

    # 12. EyedropperTool
    eyedropper = tools[ToolType.EYEDROPPER.value]
    assert isinstance(eyedropper, EyedropperTool)
    eyedropper.mouse_press(ft_doc, 25.0, 25.0, button=1)
    eyedropper.mouse_release(ft_doc, 25.0, 25.0, button=1)
    assert ctx.fg_color == (0, 128, 255, 255)

    # 13. BucketFillTool
    fill = tools[ToolType.BUCKET_FILL.value]
    assert isinstance(fill, BucketFillTool)
    fill_doc = DocumentSession(40, 40, title="FillDoc", initial_fill=(20, 20, 20, 255))
    fill_doc.active_layer.pixels[5:35, 5] = [255, 255, 255, 255]
    fill_doc.active_layer.pixels[5:35, 34] = [255, 255, 255, 255]
    fill_doc.active_layer.pixels[5, 5:35] = [255, 255, 255, 255]
    fill_doc.active_layer.pixels[34, 5:35] = [255, 255, 255, 255]
    ctx.fg_color = (255, 165, 0, 255)
    ctx.fill_tolerance = 5
    ctx.fill_contiguous = True
    fill.mouse_press(fill_doc, 20.0, 20.0, button=1)
    assert tuple(fill_doc.active_layer.pixels[20, 20]) == (255, 165, 0, 255)
    assert tuple(fill_doc.active_layer.pixels[0, 0]) == (20, 20, 20, 255)
    fill_doc.undo()
    assert tuple(fill_doc.active_layer.pixels[20, 20]) == (20, 20, 20, 255)

    # 14. GradientTool
    grad = tools[ToolType.GRADIENT.value]
    assert isinstance(grad, GradientTool)
    grad_doc = DocumentSession(64, 64, title="GradDoc")
    ctx.fg_color = (255, 0, 0, 255)
    ctx.bg_color = (0, 0, 255, 255)
    grad.mouse_press(grad_doc, 0.0, 32.0, button=1)
    grad.mouse_move(grad_doc, 63.0, 32.0, buttons=1)
    grad.mouse_release(grad_doc, 63.0, 32.0, button=1)
    assert grad_doc.active_layer.pixels[32, 0, 0] > 240
    assert grad_doc.active_layer.pixels[32, 63, 2] > 240
    assert grad_doc.active_layer.pixels[32, 32, 0] > 100 and grad_doc.active_layer.pixels[32, 32, 2] > 100
    grad_doc.undo()
    assert np.all(grad_doc.active_layer.pixels[:, :, 3] == 0)

    # 15. TextTool
    text_tool = tools[ToolType.TEXT.value]
    assert isinstance(text_tool, TextTool)
    text_doc = DocumentSession(160, 80, title="TextDoc")
    ctx.fg_color = (255, 255, 255, 255)
    ctx.font_size = 22
    ctx.text_content = "Lite"
    text_tool.mouse_press(text_doc, 10.0, 10.0, button=1)
    assert int(np.sum(text_doc.active_layer.pixels[:, :, 3] > 0)) > 20
    text_doc.undo()
    assert np.all(text_doc.active_layer.pixels[:, :, 3] == 0)


def test_main_window_canvas_panels_and_dialogs() -> None:
    import numpy as np
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QGuiApplication, QImage, QKeyEvent, QMouseEvent, QWheelEvent

    from app.core.ai_bg_remove import apply_ai_background_removal, generate_alpha_matte
    from app.core.batch import BatchOperationConfig, BatchProcessor
    from app.core.c_bridge import get_engine
    from app.core.layer import BlendMode
    from app.main import create_application, resolve_asset_path
    from app.tools import ToolType
    from app.ui.dialogs import (
        BatchProcessorDialog,
        ChromaKeyDialog,
        GridSplitterDialog,
        HotkeyEditorDialog,
        QuickExportDialog,
        SmartAlphaSplitterDialog,
        load_keybindings,
    )

    assert resolve_asset_path("app/ui/theme.qss").is_file()
    assert resolve_asset_path("app/config/keybindings.json").is_file()

    qapp, win = create_application([])
    win.show()
    qapp.processEvents()

    # Verify strict lazy loading even after MainWindow creation
    assert "onnxruntime" not in sys.modules

    # 0. Verify RU default localization, clean white background, and no '*' on untouched document
    from app.config.i18n import get_language

    assert get_language() == "ru"
    assert "Слои" in win.layers_dock.windowTitle()
    assert win.new_tab_btn.text() == "+"

    # 1. Multi-tab switching, inactive cache release, modified '*' title
    doc1 = win.active_document
    assert doc1 is not None
    assert doc1.modified is False and not doc1.display_title.endswith("*")
    assert tuple(doc1.active_layer.pixels[0, 0]) == (255, 255, 255, 255)
    doc1.composite()
    assert doc1._composited_cache is not None

    doc2 = win.new_document(400, 300, title="TabTwo", initial_fill=(20, 40, 60, 255))
    doc2.composite()
    assert doc1.is_active_tab is False and doc1._composited_cache is None
    assert doc2.is_active_tab is True and doc2._composited_cache is not None

    win.tab_bar.setCurrentIndex(0)
    qapp.processEvents()
    assert doc1.is_active_tab is True
    assert doc2.is_active_tab is False and doc2._composited_cache is None

    # Paint on doc1 to trigger '*' modified indicator in tab title
    doc1.add_layer(name="StrokeLayer")
    win._refresh_canvas_and_tabs()
    assert doc1.display_title.endswith("*")
    assert win.tab_bar.tabText(0).endswith("*")

    # 2. Clipboard Quick-Create (Ctrl+V)
    clip_img = QImage(48, 36, QImage.Format_RGBA8888)
    clip_img.fill(Qt.red)
    QGuiApplication.clipboard().setImage(clip_img)
    doc1.clear_selection()
    prev_count = len(win.tab_manager.sessions)
    clip_doc = win.paste_from_clipboard()
    assert clip_doc is not None and len(win.tab_manager.sessions) == prev_count + 1
    assert (clip_doc.width, clip_doc.height) == (48, 36)
    assert tuple(clip_doc.active_layer.pixels[10, 10]) == (255, 0, 0, 255)

    # 3. Tab-key Zen mode toggle
    assert win.is_zen_mode is False
    assert win.layers_dock.isVisible() is True
    assert win.tools_toolbar.isVisible() is True

    win.canvas_view.setFocus()
    tab_press = QKeyEvent(QEvent.KeyPress, Qt.Key_Tab, Qt.NoModifier)
    QGuiApplication.sendEvent(win.canvas_view, tab_press)
    qapp.processEvents()
    assert win.is_zen_mode is True
    assert win.layers_dock.isVisible() is False
    assert win.color_dock.isVisible() is False
    assert win.tools_toolbar.isVisible() is False
    assert win.tool_options_bar.isVisible() is False
    assert win.statusBar().isVisible() is False

    tab_press2 = QKeyEvent(QEvent.KeyPress, Qt.Key_Tab, Qt.NoModifier)
    QGuiApplication.sendEvent(win.canvas_view, tab_press2)
    qapp.processEvents()
    assert win.is_zen_mode is False
    assert win.layers_dock.isVisible() is True
    assert win.tools_toolbar.isVisible() is True
    assert win.statusBar().isVisible() is True

    # 4. CanvasView zoom-to-cursor & temporary Space panning
    cv = win.canvas_view
    win.select_tool(ToolType.BRUSH.value)
    assert cv.active_tool is win.tools[ToolType.BRUSH.value]

    cursor_pos = QPointF(180.0, 140.0)
    cx_before, cy_before = clip_doc.viewport.screen_to_canvas(cursor_pos.x(), cursor_pos.y())
    zoom_before = clip_doc.viewport.zoom
    wheel_ev = QWheelEvent(
        cursor_pos,
        cursor_pos,
        QPointF(0, 0).toPoint(),
        QPointF(0, 120).toPoint(),
        Qt.NoButton,
        Qt.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    QGuiApplication.sendEvent(cv, wheel_ev)
    assert clip_doc.viewport.zoom > zoom_before
    cx_after, cy_after = clip_doc.viewport.screen_to_canvas(cursor_pos.x(), cursor_pos.y())
    assert abs(cx_before - cx_after) < 1e-5 and abs(cy_before - cy_after) < 1e-5

    # Temporary Space + Left Drag panning restores BrushTool on release
    pan_x0, pan_y0 = clip_doc.viewport.pan_x, clip_doc.viewport.pan_y
    QGuiApplication.sendEvent(cv, QKeyEvent(QEvent.KeyPress, Qt.Key_Space, Qt.NoModifier))
    assert cv.is_temp_panning is True
    QGuiApplication.sendEvent(
        cv,
        QMouseEvent(
            QEvent.MouseButtonPress,
            QPointF(100, 100),
            QPointF(100, 100),
            Qt.LeftButton,
            Qt.LeftButton,
            Qt.NoModifier,
        ),
    )
    QGuiApplication.sendEvent(
        cv,
        QMouseEvent(
            QEvent.MouseMove,
            QPointF(140, 125),
            QPointF(140, 125),
            Qt.NoButton,
            Qt.LeftButton,
            Qt.NoModifier,
        ),
    )
    QGuiApplication.sendEvent(
        cv,
        QMouseEvent(
            QEvent.MouseButtonRelease,
            QPointF(140, 125),
            QPointF(140, 125),
            Qt.LeftButton,
            Qt.NoButton,
            Qt.NoModifier,
        ),
    )
    QGuiApplication.sendEvent(cv, QKeyEvent(QEvent.KeyRelease, Qt.Key_Space, Qt.NoModifier))
    assert (clip_doc.viewport.pan_x, clip_doc.viewport.pan_y) == (pan_x0 + 40.0, pan_y0 + 25.0)
    assert cv.active_tool is win.tools[ToolType.BRUSH.value]

    # 5. LayersPanel mask / clipping / blend-mode / thumbnail / row height actions
    lp = win.layers_panel
    lp.set_document(clip_doc)
    new_layer = lp.add_layer()
    assert new_layer is not None and len(clip_doc.layers) == 2
    assert lp.layer_list.item(0).sizeHint().height() >= 40
    row_widget = lp.layer_list.itemWidget(lp.layer_list.item(0))
    assert row_widget is not None and not row_widget.thumb_lbl.pixmap().isNull()

    mult_idx = lp.blend_combo.findData(BlendMode.MULTIPLY)
    assert mult_idx >= 0
    lp.blend_combo.setCurrentIndex(mult_idx)
    assert new_layer.blend_mode == BlendMode.MULTIPLY

    lp.opacity_slider.setValue(60)
    assert abs(new_layer.opacity - 0.60) < 1e-6

    lp.toggle_clipping_mask()
    assert new_layer.clipping_mask is True

    lp.add_or_delete_mask()
    assert new_layer.mask is not None and new_layer.mask.enabled is True
    lp.toggle_mask_enabled()
    assert new_layer.mask.enabled is False
    lp.chk_edit_mask.setChecked(True)
    assert win.tool_context.edit_mask is True
    lp.add_or_delete_mask()
    assert new_layer.mask is None and win.tool_context.edit_mask is False

    # 6. QuickExportDialog live KB/MB calculation across formats/scale/quality
    exp_doc = win.new_document(256, 256, title="ExportTest", initial_fill=(40, 120, 220, 255))
    get_engine().brush_dab(exp_doc.active_layer.pixels, 128.0, 128.0, 60.0, 0.3, (255, 220, 40, 255))
    exp_dlg = QuickExportDialog(exp_doc, parent=win)
    png_bytes = exp_dlg.estimated_bytes
    assert "KB" in exp_dlg.size_label.text() or "MB" in exp_dlg.size_label.text()
    assert png_bytes > 0

    exp_dlg.format_combo.setCurrentText("JPEG")
    jpeg_q90_bytes = exp_dlg.estimated_bytes
    exp_dlg.quality_slider.setValue(20)
    jpeg_q20_bytes = exp_dlg.estimated_bytes
    assert jpeg_q20_bytes < jpeg_q90_bytes

    exp_dlg.format_combo.setCurrentText("WEBP")
    exp_dlg.scale_slider.setValue(50)
    webp_50_bytes = exp_dlg.estimated_bytes
    assert webp_50_bytes > 0
    assert "128 x 128 px" in exp_dlg.dimensions_label.text()
    exp_dlg.scale_slider.setValue(200)
    assert exp_dlg.estimated_bytes > webp_50_bytes
    assert "512 x 512 px" in exp_dlg.dimensions_label.text()
    exp_dlg.reject()

    # 7. ChromaKeyDialog (live preview, cancel restore, accept + undo)
    ck_doc = win.new_document(40, 40, title="ChromaTest", initial_fill=(0, 255, 0, 255))
    ck_doc.active_layer.pixels[10:30, 10:30] = [220, 30, 30, 255]
    ck_cancel = ChromaKeyDialog(ck_doc, parent=win)
    assert ck_doc.active_layer.pixels[0, 0, 3] == 0  # live preview applied
    ck_cancel.reject()
    assert ck_doc.active_layer.pixels[0, 0, 3] == 255  # restored on cancel

    ck_accept = ChromaKeyDialog(ck_doc, parent=win)
    ck_accept.tolerance_slider.setValue(50)
    ck_accept.accept()
    assert ck_doc.active_layer.pixels[0, 0, 3] == 0
    assert ck_doc.active_layer.pixels[20, 20, 3] == 255
    ck_doc.undo()
    assert ck_doc.active_layer.pixels[0, 0, 3] == 255
    ck_doc.redo()
    assert ck_doc.active_layer.pixels[0, 0, 3] == 0

    # 8. GridSplitter & SmartAlphaSplitter (tabs + PNG export)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        grid_doc = win.new_document(64, 64, title="GridDoc", initial_fill=(10, 20, 30, 255))
        grid_dlg = GridSplitterDialog(grid_doc, win.tab_manager, parent=win)
        grid_dlg.rows_spin.setValue(2)
        grid_dlg.cols_spin.setValue(2)
        # Interactive exclusion & ZIP archive export
        grid_dlg.toggle_slice_excluded(1)
        assert len(grid_dlg.compute_slices()) == 3
        assert "3" in grid_dlg.selection_count_label.text()
        grid_zip = grid_dlg.export_to_zip(tmp_path / "grid_slices.zip")
        assert grid_zip is not None and grid_zip.is_file()
        import zipfile
        with zipfile.ZipFile(grid_zip, "r") as zf:
            assert len(zf.namelist()) == 3
        grid_dlg.toggle_slice_excluded(1)  # re-include all 4
        grid_files = grid_dlg.export_to_folder(tmp_path / "grid_out")
        assert len(grid_files) == 4 and all(p.is_file() for p in grid_files)
        grid_tabs = grid_dlg.open_in_tabs()
        assert len(grid_tabs) == 4 and all((d.width, d.height) == (32, 32) for d in grid_tabs)
        grid_dlg.reject()

        sprite_doc = win.new_document(64, 64, title="SpriteDoc", initial_fill=(0, 0, 0, 0))
        sprite_doc.active_layer.pixels[4:16, 4:16] = [255, 0, 0, 255]
        sprite_doc.active_layer.pixels[32:48, 36:52] = [0, 255, 0, 255]
        previewed: list = []
        alpha_dlg = SmartAlphaSplitterDialog(
            sprite_doc,
            win.tab_manager,
            on_preview_boxes=previewed.append,
            parent=win,
        )
        assert len(alpha_dlg.detected_boxes) == 2
        assert len(previewed[-1]) == 2
        alpha_dlg.toggle_sprite_excluded(0)
        assert len(alpha_dlg.extract_slices()) == 1
        alpha_zip = alpha_dlg.export_to_zip(tmp_path / "alpha_sprites.zip")
        assert alpha_zip is not None and alpha_zip.is_file()
        with zipfile.ZipFile(alpha_zip, "r") as zf:
            assert len(zf.namelist()) == 1
        alpha_dlg.toggle_sprite_excluded(0)  # re-include both
        alpha_files = alpha_dlg.export_to_folder(tmp_path / "alpha_out")
        assert len(alpha_files) == 2 and all(p.is_file() for p in alpha_files)
        alpha_tabs = alpha_dlg.open_in_tabs()
        assert len(alpha_tabs) == 2
        alpha_dlg.reject()

        # 9. BatchProcessor + BatchProcessorDialog
        from PIL import Image

        batch_in = tmp_path / "batch_in"
        batch_out = tmp_path / "batch_out"
        batch_in.mkdir()
        img_arr = np.zeros((32, 32, 4), dtype=np.uint8)
        img_arr[:, :] = [0, 255, 0, 255]
        img_arr[8:24, 8:24] = [255, 50, 50, 255]
        Image.fromarray(img_arr, mode="RGBA").save(batch_in / "sample_a.png")
        Image.fromarray(img_arr, mode="RGBA").save(batch_in / "sample_b.png")

        cfg = BatchOperationConfig(
            input_dir=batch_in,
            output_dir=batch_out,
            output_format="WEBP",
            quality=85,
            chroma_key_enabled=True,
            chroma_key_rgb=(0, 255, 0),
            trim_transparent=True,
            resize_width=32,
            resize_height=32,
            max_workers=2,
        )
        progress_events: list[tuple[int, int, str]] = []
        res = BatchProcessor(cfg).run(progress_callback=lambda d, t, n: progress_events.append((d, t, n)))
        assert res.total == 2 and res.succeeded == 2 and res.failed == 0
        assert len(progress_events) == 2
        assert all(p.is_file() and p.suffix == ".webp" for p in res.output_paths)

        batch_dlg_out = tmp_path / "batch_dlg_out"
        batch_dlg = BatchProcessorDialog(parent=win)
        batch_dlg.input_dir_edit.setText(str(batch_in))
        batch_dlg.output_dir_edit.setText(str(batch_dlg_out))
        batch_dlg.format_combo.setCurrentText("PNG")
        batch_dlg.chroma_key_check.setChecked(True)
        batch_dlg.trim_check.setChecked(True)
        dlg_res = batch_dlg.run_batch_sync()
        assert dlg_res.succeeded == 2 and dlg_res.failed == 0
        batch_dlg.reject()

        # 10. HotkeyEditorDialog reading/saving app/config/keybindings.json
        kb_path = PROJECT_ROOT / "app" / "config" / "keybindings.json"
        live_bindings = load_keybindings(kb_path)
        assert live_bindings["brush"] == "B" and live_bindings["zen_mode"] == "Tab"

        default_hk_dlg = HotkeyEditorDialog(parent=win)
        assert default_hk_dlg.get_bindings()["brush"] == "B"
        default_hk_dlg.reject()

        # Round-trip save via temporary copy and live path verification
        tmp_kb = tmp_path / "keybindings.json"
        tmp_kb.write_text(kb_path.read_text(encoding="utf-8"), encoding="utf-8")
        hk_dlg = HotkeyEditorDialog(keybindings_path=tmp_kb, parent=win)
        hk_dlg.set_binding("brush", "Alt+B")
        hk_dlg.accept()
        reloaded = load_keybindings(tmp_kb)
        assert reloaded["brush"] == "Alt+B"

        hk_dlg2 = HotkeyEditorDialog(keybindings_path=tmp_kb, parent=win)
        hk_dlg2.reset_defaults()
        hk_dlg2.accept()
        assert load_keybindings(tmp_kb)["brush"] == "B"

    # 11. Lazy AI background removal offline fallback check (still zero onnxruntime import until real ONNX run)
    ai_doc = win.new_document(40, 40, title="AITest", initial_fill=(0, 255, 0, 255))
    ai_doc.active_layer.pixels[10:30, 10:30] = [240, 30, 30, 255]
    missing_model = Path(tempfile.gettempdir()) / "nonexistent_u2netp_test.onnx"
    matte = generate_alpha_matte(
        ai_doc.active_layer.pixels,
        model_path=missing_model,
        allow_download=False,
        offline_fallback=True,
    )
    assert matte.shape == (40, 40) and matte[20, 20] > 200 and matte[0, 0] < 20
    mask = apply_ai_background_removal(
        ai_doc,
        model_path=missing_model,
        allow_download=False,
        offline_fallback=True,
    )
    assert ai_doc.active_layer.mask is mask
    ai_doc.undo()
    assert ai_doc.active_layer.mask is None
    assert "onnxruntime" not in sys.modules

    # 12. Vectorized CanvasView selection rendering at zoom < 1 + Crop UI presets + Drag&Drop + Clipboard URL + i18n switch
    from PySide6.QtCore import QMimeData, QUrl
    from PySide6.QtGui import QDropEvent

    sel_mask = np.zeros((doc1.height, doc1.width), dtype=np.uint8)
    sel_mask[100:180, 100:220] = 255
    sel_mask[110:150, 120:160] = 0  # Non-rectangular hollow selection
    doc1.set_selection_mask(sel_mask)
    doc1.viewport.zoom = 0.4
    win.switch_to_tab(0)
    win.canvas_view._invalidate_selection_cache()
    win.canvas_view.repaint()
    qapp.processEvents()
    assert win.canvas_view._sel_segments_np is not None and win.canvas_view._sel_segments_np.shape[0] >= 8
    doc1.clear_selection()

    # Crop UI preset & swap orientation
    win.select_tool(ToolType.CROP.value)
    idx_169 = win.tool_options_bar.crop_ratio_combo.findData("16:9")
    assert idx_169 >= 0
    win.tool_options_bar.crop_ratio_combo.setCurrentIndex(idx_169)
    assert win.tool_context.crop_aspect_preset == "16:9"
    win.tool_options_bar.btn_crop_swap.click()
    assert win.tool_context.crop_aspect_preset == "9:16"
    win.tools[ToolType.CROP.value].cancel_crop()

    # Drag & Drop onto Canvas (new centered layer with Undo) and onto MainWindow (new tab) + Explorer Ctrl+V URL paste
    with tempfile.TemporaryDirectory() as dnd_dir:
        from PIL import Image as _PILImage

        drop_img_path = Path(dnd_dir) / "dropped_sprite.png"
        arr = np.full((24, 32, 4), [0, 200, 255, 255], dtype=np.uint8)
        _PILImage.fromarray(arr, mode="RGBA").save(drop_img_path)

        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(str(drop_img_path))])

        # Drop onto CanvasView -> inserts centered layer
        prev_layers = len(doc1.layers)
        drop_ev = QDropEvent(QPointF(50, 50), Qt.DropAction.CopyAction, mime, Qt.MouseButton.LeftButton, Qt.NoModifier)
        win.canvas_view.dropEvent(drop_ev)
        assert len(doc1.layers) == prev_layers + 1
        assert doc1.active_layer.name == "dropped_sprite"
        doc1.undo()
        assert len(doc1.layers) == prev_layers

        # Drop onto MainWindow -> opens new tab
        prev_tabs = len(win.tab_manager.sessions)
        win_drop_ev = QDropEvent(QPointF(10, 10), Qt.DropAction.CopyAction, mime, Qt.MouseButton.LeftButton, Qt.NoModifier)
        win.dropEvent(win_drop_ev)
        assert len(win.tab_manager.sessions) == prev_tabs + 1
        assert win.active_document.title == "dropped_sprite.png"
        assert win.active_document.modified is False

        # Clipboard file URL paste (Explorer Ctrl+C -> Ctrl+V)
        url_mime = QMimeData()
        url_mime.setUrls([QUrl.fromLocalFile(str(drop_img_path))])
        QGuiApplication.clipboard().setMimeData(url_mime)
        target_doc = win.active_document
        layers_before_paste = len(target_doc.layers)
        win.paste_from_clipboard()
        assert len(target_doc.layers) == layers_before_paste + 1
        QGuiApplication.clipboard().clear()

    # Language switch RU -> EN -> RU (non-persistent in test)
    win.set_language("en", persist=False)
    assert "Layers" in win.layers_dock.windowTitle()
    win.set_language("ru", persist=False)
    assert "Слои" in win.layers_dock.windowTitle()

    # 13. Canvas RMB Context Menu & Selection Actions (Ctrl+J, Ctrl+Shift+J, Invert, Fill, Clear, Crop)
    ctx_doc = win.new_document(60, 60, title="CtxMenuDoc", initial_fill=(100, 150, 200, 255))
    menu_no_sel = win.build_canvas_context_menu()
    assert len(menu_no_sel.actions()) == 4

    sel = np.zeros((60, 60), dtype=np.uint8)
    sel[10:30, 10:30] = 255
    ctx_doc.set_selection_mask(sel)
    menu_with_sel = win.build_canvas_context_menu()
    assert len(menu_with_sel.actions()) >= 8

    # Invert selection
    win.invert_selection()
    assert ctx_doc.selection_mask[20, 20] == 0 and ctx_doc.selection_mask[5, 5] == 255
    win.invert_selection()
    assert ctx_doc.selection_mask[20, 20] == 255

    # Layer via Copy (Ctrl+J)
    copied_layer = win.layer_via_copy()
    assert copied_layer is not None and len(ctx_doc.layers) == 2
    assert (copied_layer.offset_x, copied_layer.offset_y) == (10, 10)
    assert (copied_layer.width, copied_layer.height) == (20, 20)
    assert ctx_doc.selection_mask is None

    # Layer via Cut (Ctrl+Shift+J) + Undo
    ctx_doc.set_active_layer(0)
    ctx_doc.set_selection_mask(sel)
    cut_layer = win.layer_via_cut()
    assert cut_layer is not None and len(ctx_doc.layers) == 3
    assert ctx_doc.layers[0].pixels[20, 20, 3] == 0
    ctx_doc.undo()
    assert len(ctx_doc.layers) == 2 and ctx_doc.layers[0].pixels[20, 20, 3] == 255

    # Fill selection & Clear selection contents
    ctx_doc.set_active_layer(0)
    ctx_doc.set_selection_mask(sel)
    win.tool_context.fg_color = (255, 10, 20, 255)
    assert win.fill_selection_with_fg() is True
    assert tuple(ctx_doc.layers[0].pixels[20, 20]) == (255, 10, 20, 255)
    assert win.delete_selection_contents() is True
    assert ctx_doc.layers[0].pixels[20, 20, 3] == 0
    ctx_doc.undo()
    assert tuple(ctx_doc.layers[0].pixels[20, 20]) == (255, 10, 20, 255)

    # Crop to selection + Undo
    assert win.crop_to_selection() is True
    assert (ctx_doc.width, ctx_doc.height) == (20, 20)
    ctx_doc.undo()
    assert (ctx_doc.width, ctx_doc.height) == (60, 60)

    win.close()
    qapp.processEvents()


if __name__ == "__main__":
    test_imports_and_lazy_loading()
    test_all_15_tools()
    test_main_window_canvas_panels_and_dialogs()
    print("ALL FULL INTEGRATION SMOKE TESTS PASSED")
