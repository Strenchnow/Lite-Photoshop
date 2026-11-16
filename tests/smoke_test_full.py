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

    # 8. LassoTool
    lasso = tools[ToolType.LASSO.value]
    assert isinstance(lasso, LassoTool)
    lasso.mouse_press(sel_doc, 10.0, 10.0, button=1)
    lasso.mouse_move(sel_doc, 50.0, 10.0, buttons=1)
    lasso.mouse_move(sel_doc, 30.0, 50.0, buttons=1)
    lasso.mouse_release(sel_doc, 10.0, 10.0, button=1)
    assert sel_doc.selection_mask is not None
    assert sel_doc.selection_mask[25, 30] == 255
    assert sel_doc.selection_mask[5, 5] == 0
    sel_doc.clear_selection()

    # 9. MagicWandTool
    wand = tools[ToolType.MAGIC_WAND.value]
    assert isinstance(wand, MagicWandTool)
    sel_doc.active_layer.pixels[20:40, 20:40] = [10, 200, 220, 255]
    ctx.wand_tolerance = 10
    ctx.wand_contiguous = True
    wand.mouse_press(sel_doc, 25.0, 25.0, button=1)
    assert sel_doc.selection_mask is not None
    assert int(np.sum(sel_doc.selection_mask == 255)) == 400
    sel_doc.clear_selection()

    # 10. CropTool (Esc cancel, Enter apply + undo)
    crop = tools[ToolType.CROP.value]
    assert isinstance(crop, CropTool)
    crop_doc = DocumentSession(100, 100, title="CropDoc")
    crop_doc.active_layer.pixels[20:40, 20:40] = [255, 0, 255, 255]
    crop.mouse_press(crop_doc, 15.0, 15.0, button=1)
    crop.mouse_move(crop_doc, 55.0, 45.0, buttons=1)
    crop.mouse_release(crop_doc, 55.0, 45.0, button=1)
    assert crop.crop_rect == (15, 15, 40, 30)
    assert crop.key_press(crop_doc, Qt.Key_Escape) is True
    assert crop.crop_rect is None and (crop_doc.width, crop_doc.height) == (100, 100)

    crop.mouse_press(crop_doc, 15.0, 15.0, button=1)
    crop.mouse_release(crop_doc, 55.0, 45.0, button=1)
    assert crop.key_press(crop_doc, Qt.Key_Return) is True
    assert (crop_doc.width, crop_doc.height) == (40, 30)
    assert (crop_doc.active_layer.offset_x, crop_doc.active_layer.offset_y) == (-15, -15)
    crop_doc.undo()
    assert (crop_doc.width, crop_doc.height) == (100, 100)

    # 11. FreeTransformTool (aspect-lock Shift drag + rotation + Enter commit + undo)
    ft = tools[ToolType.TRANSFORM.value]
    assert isinstance(ft, FreeTransformTool)
    ft_doc = DocumentSession(120, 120, title="TransformDoc")
    ft_doc.active_layer.pixels[20:40, 20:40] = [0, 128, 255, 255]
    ft.on_activate(ft_doc)
    assert ft.active is True
    assert (ft.orig_w, ft.orig_h) == (20.0, 20.0)
    # Drag SE corner handle (40, 40) to (70, 55) with MOD_SHIFT -> aspect ratio stays 1:1
    ft.mouse_press(ft_doc, 40.0, 40.0, button=1)
    ft.mouse_move(ft_doc, 70.0, 55.0, buttons=1, modifiers=MOD_SHIFT)
    ft.mouse_release(ft_doc, 70.0, 55.0, button=1, modifiers=MOD_SHIFT)
    assert abs(ft.box_w - ft.box_h) < 1e-4 and ft.box_w > 30.0
    assert ft.key_press(ft_doc, Qt.Key_Escape) is True
    assert ft.active is False

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

    # 1. Multi-tab switching, inactive cache release, modified '*' title
    doc1 = win.active_document
    assert doc1 is not None
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

    # 5. LayersPanel mask / clipping / blend-mode actions
    lp = win.layers_panel
    lp.set_document(clip_doc)
    new_layer = lp.add_layer()
    assert new_layer is not None and len(clip_doc.layers) == 2

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
        grid_files = grid_dlg.export_to_folder(tmp_path / "grid_out")
        assert len(grid_files) == 4 and all(p.is_file() for p in grid_files)
        grid_tabs = grid_dlg.open_in_tabs()
        assert len(grid_tabs) == 4 and all((d.width, d.height) == (32, 32) for d in grid_tabs)
        grid_dlg.reject()

        sprite_doc = win.new_document(64, 64, title="SpriteDoc")
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

    # 11. Lazy AI background removal offline fallback check (still zero onnxruntime import)
    ai_doc = win.new_document(40, 40, title="AITest", initial_fill=(0, 255, 0, 255))
    ai_doc.active_layer.pixels[10:30, 10:30] = [240, 30, 30, 255]
    matte = generate_alpha_matte(ai_doc.active_layer.pixels, allow_download=False, offline_fallback=True)
    assert matte.shape == (40, 40) and matte[20, 20] > 200 and matte[0, 0] < 20
    mask = apply_ai_background_removal(ai_doc, allow_download=False, offline_fallback=True)
    assert ai_doc.active_layer.mask is mask
    ai_doc.undo()
    assert ai_doc.active_layer.mask is None
    assert "onnxruntime" not in sys.modules

    win.close()
    qapp.processEvents()


if __name__ == "__main__":
    test_imports_and_lazy_loading()
    test_all_15_tools()
    test_main_window_canvas_panels_and_dialogs()
    print("ALL FULL INTEGRATION SMOKE TESTS PASSED")
