# Project Specification & Master Prompt: LitePhotoEditor

## Role & Objective
Act as a Senior Systems & Graphics Software Engineer. Build a lightweight, high-performance, Photoshop-inspired raster image editor for Windows using a hybrid **C + Python** architecture, designed to be compiled into a standalone `.exe` binary.

- **Memory Footprint Target:** < 80 MB RAM at idle (with an open empty canvas).
- **Zero-Lag Viewport:** Smooth 60+ FPS zooming and panning even on 4K canvases.
- **Strict Lazy Loading:** Heavy dependencies (such as the ONNX AI background removal runtime and model) must ONLY be imported/loaded into memory when triggered by the user, and explicitly unloaded (`del session; gc.collect()`) immediately after execution.

---

## Technology Stack
1. **UI & Application Shell:** Python 3.11+ with **PySide6** (Qt6), styled with a clean, dark, low-distraction Photoshop/Figma-inspired QSS theme (`theme.qss`).
2. **Core Pixel & Math Engine:** Pure **C99/C11** compiled into a Windows shared library (`core_engine.dll`) using `gcc -O3 -shared -fopenmp` (with automatic fallback detection for `clang` or MSVC `cl.exe`).
3. **Python-C Interop:** `ctypes` passing raw data pointers directly from contiguous `NumPy` (`uint8` RGBA) arrays — **zero buffer copying** between Python and C.
4. **AI Segmentation:** `onnxruntime` (CPU execution provider) + a lightweight segmentation model (`u2netp.onnx` ~4.5 MB, auto-downloaded on first use into `assets/models/` if missing).
5. **Packaging:** `PyInstaller` (`build_exe.py`) packaging the entire application into a clean standalone Windows `.exe`, properly bundling `core_engine.dll`, QSS styles, icons, and `keybindings.json`.

---

## Directory Architecture
Generate the project using this exact modular folder layout:

- `LitePhotoEditor/`
  - `c_core/`
    - `core_engine.h` — C API declarations (`__declspec(dllexport)`)
    - `core_engine.c` — Compositing, blend modes, filters, smudge, chroma key, flood fill, connected components
    - `compile_dll.py` — Cross-compiler detection & build helper (`gcc` / `clang` / `cl.exe`)
  - `app/`
    - `main.py` — Entry point & `QApplication` setup
    - `core/`
      - `c_bridge.py` — `ctypes` bindings & automatic DLL compilation check
      - `document.py` — `DocumentSession` (layers, delta undo/redo, viewport transform matrix)
      - `layer.py` — `Layer` & `LayerMask` models (`RGBA uint8` NumPy buffers)
      - `ai_bg_remove.py` — Lazy-loaded ONNX background removal + immediate RAM cleanup
      - `splitter.py` — Grid Splitter & Smart Alpha Connected-Component Splitter
      - `batch.py` — Multi-threaded folder batch processor
    - `tools/`
      - `base_tool.py` — Abstract tool interface (`mousePress`, `mouseMove`, `mouseRelease`, `keyPress`)
      - `brush_tools.py` — Brush (`B`), Eraser (`E`), Smudge, Local Blur/Sharpen Brush
      - `nav_tools.py` — Hand Tool (`H`), Move Tool (`V`) with magnetic snapping
      - `select_tools.py` — Rect Marquee (`M`), Lasso (`L`), Magic Wand (`W`), Crop (`C`), Free Transform (`Ctrl+T`)
      - `utility_tools.py` — Eyedropper (`I` / `Alt`), Bucket Fill (`G`), Linear Gradient, Text Tool (`T`)
    - `ui/`
      - `main_window.py` — Tabbed workspace, dockable panels, `Tab`-key Zen mode
      - `canvas_view.py` — High-FPS viewport, Zoom-to-Cursor math, snapping guide lines overlay
      - `panels.py` — Layers stack, Tool options bar, Color picker & swatches
      - `dialogs.py` — Quick Export (live KB size preview), Grid & Alpha Splitters, Batch Processor, Hotkey Editor
      - `theme.qss` — Modern dark stylesheet
    - `config/`
      - `keybindings.json` — User-customizable hotkeys
  - `assets/` — Vector/PNG toolbar icons & `models/` cache folder
  - `build_exe.py` — Full pipeline: compile C DLL -> package `.exe` via PyInstaller
  - `requirements.txt`
  - `SPEC.md`

---

## Detailed Functional Specifications

### 1. Multi-Tab Workspace & Memory Efficiency
- **Tabbed Document Interface (`QTabWidget`):** Support multiple open documents simultaneously. Display an unsaved indicator (`*`) in the tab title when modified.
- **Zero-Overhead Inactive Tabs:** Only the currently active tab maintains a composited display image/texture. Background tabs store only raw layer buffers and document state, consuming zero GPU/display overhead.
- **Clipboard Quick-Create (`Ctrl+V`):** When pressing `Ctrl+V` with no active internal pixel selection, inspect the Windows clipboard (`QGuiApplication.clipboard().image()`). If an image is present, immediately create a new tab with canvas dimensions matching the clipboard image and paste it as the base layer.
- **Delta Undo/Redo System:** Instead of cloning full canvas buffers on every edit, compute the bounding box (`dirty_rect`) of modified pixels before and after an action, storing only the cropped patch in the undo stack (30+ steps with minimal RAM impact).

### 2. Viewport Navigation, Magnetic Snapping & Alignment
- **Zoom-to-Cursor:** Mouse wheel scrolling must zoom smoothly while keeping the exact canvas coordinate under the mouse cursor stationary on screen via 2D affine view-to-world matrices.
- **Hand Tool (`H`) & Panning:** Dedicated Hand tool (`H`) to drag the viewport. Holding `Space + Left Mouse Drag` or `Middle Mouse Drag` from *any* active tool must temporarily activate Hand panning and restore the previous tool upon release.
- **Smart Snapping & Guides:**
  - While dragging a layer with the Move tool (`V`), automatically snap the layer's bounding box to the **canvas edges** (top, bottom, left, right) and **canvas center axes** (horizontal center, vertical center) within a 6-pixel screen threshold.
  - Render crisp 1px magenta/cyan alignment guide lines across the canvas whenever snapping engages.
  - **Free Move Override (`Shift`):** Holding `Shift` while dragging temporarily bypasses all snapping rules for unrestricted pixel placement.
- **1-Click Centering:** Provide toolbar buttons and shortcuts to center the active layer horizontally, vertically, or dead-center on the canvas.

### 3. Layers, Masks & C-Accelerated Compositing
- **Layer Operations:** Create, delete, duplicate, reorder (drag-and-drop or up/down buttons), rename, toggle visibility, lock layer (`Lock`), adjust Opacity (`0–100%`), and `Merge Down`.
- **Blend Modes (Implemented in C):** `Normal`, `Multiply`, `Screen`, `Overlay`, and `Add` (Linear Dodge).
- **Layer Masks & Clipping Masks:**
  - Attach a grayscale (`uint8`) raster mask to any layer for non-destructive hiding/revealing with brushes.
  - Support **Clipping Masks** (clipping a layer's alpha channel to the non-transparent pixels of the base layer directly beneath it).

### 4. Drawing, Retouch & Selection Tools
- **Brush (`B`) & Eraser (`E`):** Adjustable size, hardness (radial falloff), opacity, and flow. Interpolate stamp positions along mouse movement vectors so fast strokes remain continuous without gaps.
- **Smudge & Local Blur/Sharpen Brush (C-Accelerated):**
  - **Smudge:** Carries and blends underlying pixel colors along the brush drag direction.
  - **Local Blur / Sharpen:** Applies a localized 3x3 or 5x5 Gaussian blur or unsharp-mask kernel strictly within the brush radius.
- **Selection & Transform:**
  - Rectangular Marquee (`M`), Freehand Lasso (`L`), and Magic Wand (`W` — C flood-fill selection with adjustable `Tolerance`).
  - **Free Transform (`Ctrl+T`):** Interactive bounding box with corner/edge scale handles (`Shift` locks aspect ratio) and rotation handle, committed via bilinear/bicubic resampling.
  - **Crop Tool (`C`):** Interactive crop rectangle with `Enter` to apply and `Esc` to cancel.
- **Utility Tools:** Eyedropper (`I`, also accessible by holding `Alt` while using Brush/Fill), Bucket Fill (`G` with Tolerance), Linear Gradient Tool, and Text Tool (`T` with font family, size, weight, and color controls).

### 5. AI Removal, Chroma Key & Smart Image Splitters
- **1-Click AI Background Removal:** Loads `u2netp.onnx` lazily via `onnxruntime`, generates a soft alpha matte, applies it as a non-destructive Layer Mask on the active layer, and unloads the ONNX session immediately (`del session; gc.collect()`).
- **Chroma Key (Real-Time C Filter):** Dialog with eyedropper color selector, `Tolerance` slider, and `Edge Softness / Despill` slider that keys out solid backgrounds in real time.
- **Image Splitter (Grid Mode):** Slice the canvas or active layer into uniform `Rows x Columns` or fixed pixel tiles (e.g., `32x32`, `64x64`) and batch-export to a target folder.
- **Smart Alpha Splitter (Connected Component Labeling in C):**
  - Uses a fast C BFS / Union-Find algorithm on the alpha channel (`alpha > threshold`) to automatically detect all isolated non-transparent sprites/islands on a layer.
  - Computes tight bounding boxes with user-configurable `Padding (px)` and `Min Area` noise filter.
  - Overlays numbered bounding boxes on the canvas for preview, allowing the user to **export all detected sprites as individual PNGs** OR **open each sprite directly into a new editor tab**.
- **Trim Transparent Pixels:** 1-click crop removing empty transparent margins around the active layer or entire document.

### 6. Productivity, Batch Processing & Export
- **Quick Export Dialog (`Ctrl+Shift+E`):** Format selector (`PNG`, `JPEG`, `WEBP`), quality slider (`1–100`), scale multiplier (`25%–200%`), live preview, and **real-time calculated output file size in KB/MB** updated as sliders move.
- **Batch Processor:** Dialog to select an input folder, output folder, output format, and a customizable pipeline of operations (e.g., *AI Remove Background -> Trim Transparent Pixels -> Resize to WxH -> Save as WebP*).
- **Zen Mode (`Tab`) & Custom Hotkeys:** Pressing `Tab` hides/restores all sidebars, toolbars, and status bars for full-screen canvas work. All keyboard shortcuts are stored in `app/config/keybindings.json` and configurable through an in-app Hotkey Settings dialog.

---

## Step-by-Step Execution Plan
Execute the build in this exact sequence and verify each stage before proceeding:
1. **C Core (`c_core/`):** Write `core_engine.h`, `core_engine.c`, and `compile_dll.py`. Compile `core_engine.dll` and verify all exported symbols.
2. **Bridge & Document Model (`app/core/`):** Implement `c_bridge.py`, `layer.py`, and `document.py` with Delta Undo/Redo and multi-tab state isolation.
3. **Viewport & Tools (`app/ui/canvas_view.py`, `app/tools/`):** Build the Zoom-to-Cursor viewport, Hand tool (`H` / `Space`), Move tool (`V`) with `Shift`-overridable magnetic snapping, and all drawing/retouch brushes.
4. **Main Window & Panels (`app/ui/`):** Assemble the tabbed main window, Layers panel (masks, clipping masks, blend modes), Tool options bar, Color swatches, and `Tab`-key Zen mode.
5. **Advanced Features:** Implement `ai_bg_remove.py` (lazy ONNX), Chroma Key dialog, Grid & Smart Alpha Splitters, `Quick Export` (live KB preview), `Batch Processor`, and `keybindings.json` editor.
6. **Packaging (`build_exe.py`):** Create and test the standalone Windows `.exe` build script using `PyInstaller`, ensuring `core_engine.dll` and assets load reliably relative to `sys._MEIPASS`.