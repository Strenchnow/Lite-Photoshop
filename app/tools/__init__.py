"""Tool system exports and factory for all 15 editor tools."""

from __future__ import annotations

from typing import Optional

from app.tools.base_tool import (
    MOD_ALT,
    MOD_CTRL,
    MOD_NONE,
    MOD_SHIFT,
    BaseTool,
    ToolContext,
    ToolType,
    extract_button,
    extract_key,
    extract_modifiers,
)
from app.tools.brush_tools import (
    BlurSharpenTool,
    BrushTool,
    EraserTool,
    SmudgeTool,
)
from app.tools.nav_tools import (
    HandTool,
    MoveTool,
    center_both,
    center_horizontal,
    center_vertical,
)
from app.tools.select_tools import (
    CropTool,
    FreeTransformTool,
    LassoTool,
    MagicWandTool,
    RectMarqueeTool,
)
from app.tools.utility_tools import (
    BucketFillTool,
    EyedropperTool,
    GradientTool,
    TextTool,
)


def create_default_tools(context: Optional[ToolContext] = None) -> dict[str, BaseTool]:
    """Instantiate and return all 15 default tools keyed by ToolType string value."""
    ctx = context if context is not None else ToolContext()
    tool_instances: tuple[BaseTool, ...] = (
        MoveTool(ctx),
        HandTool(ctx),
        BrushTool(ctx),
        EraserTool(ctx),
        SmudgeTool(ctx),
        BlurSharpenTool(ctx),
        RectMarqueeTool(ctx),
        LassoTool(ctx),
        MagicWandTool(ctx),
        CropTool(ctx),
        FreeTransformTool(ctx),
        EyedropperTool(ctx),
        BucketFillTool(ctx),
        GradientTool(ctx),
        TextTool(ctx),
    )
    return {tool.tool_id: tool for tool in tool_instances}


__all__ = [
    "MOD_NONE",
    "MOD_SHIFT",
    "MOD_CTRL",
    "MOD_ALT",
    "extract_modifiers",
    "extract_button",
    "extract_key",
    "ToolType",
    "ToolContext",
    "BaseTool",
    "BrushTool",
    "EraserTool",
    "SmudgeTool",
    "BlurSharpenTool",
    "HandTool",
    "MoveTool",
    "center_horizontal",
    "center_vertical",
    "center_both",
    "RectMarqueeTool",
    "LassoTool",
    "MagicWandTool",
    "CropTool",
    "FreeTransformTool",
    "EyedropperTool",
    "BucketFillTool",
    "GradientTool",
    "TextTool",
    "create_default_tools",
]
