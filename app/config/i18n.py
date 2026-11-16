"""Bilingual (RU / EN) localization dictionaries and runtime language switcher."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable, Optional


SUPPORTED_LANGUAGES: tuple[str, ...] = ("ru", "en")
DEFAULT_LANGUAGE: str = "ru"


_TRANSLATIONS: dict[str, dict[str, str]] = {
    "ru": {
        # Window & Menus
        "app.title": "LitePhotoEditor",
        "menu.file": "&Файл",
        "menu.edit": "&Правка",
        "menu.image": "&Изображение",
        "menu.layer": "&Слой",
        "menu.tools": "&Инструменты",
        "menu.view": "&Вид",
        "menu.language": "Я&зык (Language)",
        "lang.ru": "Русский (RU)",
        "lang.en": "English (EN)",

        # File menu
        "action.new_tab": "&Новый документ",
        "action.new_custom": "Создать &холст...",
        "action.open_file": "&Открыть...",
        "action.save_file": "&Сохранить",
        "action.quick_export": "Быстрый &экспорт...",
        "action.batch": "&Пакетная обработка...",
        "action.close_tab": "&Закрыть вкладку",
        "action.exit": "В&ыход",

        # Edit menu
        "action.undo": "&Отменить",
        "action.redo": "&Повторить",
        "action.copy": "&Копировать выделение",
        "action.paste": "&Вставить / Новый документ из буфера",
        "action.deselect": "&Снять выделение",
        "action.hotkeys": "&Горячие клавиши...",

        # Image menu
        "action.trim": "&Обрезать прозрачные края",
        "action.ai_bg": "&ИИ-удаление фона",
        "action.chroma_key": "&Хромакей (удаление цвета)...",
        "action.grid_split": "Нарезка по &сетке...",
        "action.alpha_split": "Умная нарезка по &альфа-каналу...",

        # Layer menu
        "action.add_layer": "&Добавить слой",
        "action.dup_layer": "&Дублировать слой",
        "action.del_layer": "&Удалить слой",
        "action.merge_down": "&Объединить с предыдущим",
        "action.toggle_mask_add_del": "Добавить / удалить &маску слоя",
        "action.toggle_clip": "О&бтравочная маска",
        "action.center_h": "Выровнять по центру (&горизонталь)",
        "action.center_v": "Выровнять по центру (&вертикаль)",
        "action.center_both": "Выровнять по &центру холста",

        # Tools menu extra
        "action.swap_colors": "Поменять основной/фоновый цвет",
        "action.reset_colors": "Сбросить цвета (Ч/Б)",

        # View menu
        "action.zoom_in": "&Увеличить",
        "action.zoom_out": "У&меньшить",
        "action.fit_screen": "&По размеру экрана",
        "action.actual_size": "&Реальный размер (100%)",
        "action.zen_mode": "Режим &чистого холста (Zen)",

        # Tool short labels & full names
        "tool.short.move": "Перем.",
        "tool.short.hand": "Рука",
        "tool.short.marquee": "Область",
        "tool.short.lasso": "Лассо",
        "tool.short.magic_wand": "Палочка",
        "tool.short.crop": "Кадр",
        "tool.short.transform": "Трансф.",
        "tool.short.brush": "Кисть",
        "tool.short.eraser": "Ластик",
        "tool.short.smudge": "Палец",
        "tool.short.blur_sharpen": "Размытие",
        "tool.short.bucket_fill": "Заливка",
        "tool.short.gradient": "Градиент",
        "tool.short.eyedropper": "Пипетка",
        "tool.short.text": "Текст",

        "tool.name.move": "Перемещение",
        "tool.name.hand": "Рука",
        "tool.name.marquee": "Прямоугольная область",
        "tool.name.lasso": "Лассо",
        "tool.name.magic_wand": "Волшебная палочка",
        "tool.name.crop": "Рамка (Кадрирование)",
        "tool.name.transform": "Свободное трансформирование",
        "tool.name.brush": "Кисть",
        "tool.name.eraser": "Ластик",
        "tool.name.smudge": "Палец (Размазывание)",
        "tool.name.blur_sharpen": "Размытие / Резкость",
        "tool.name.bucket_fill": "Заливка",
        "tool.name.gradient": "Градиент",
        "tool.name.eyedropper": "Пипетка",
        "tool.name.text": "Текст",

        # Docks & Tabs
        "dock.colors": "Цвета и образцы",
        "dock.layers": "Слои",
        "dock.tools": "Инструменты",
        "tab.new_btn_tip": "Создать новую вкладку (Ctrl+N)",
        "doc.untitled": "Без названия",
        "doc.clipboard_prefix": "Буфер",
        "doc.layer_pasted": "Вставленный слой",
        "doc.background_layer": "Фон",

        # Canvas & Status Bar
        "canvas.empty_hint": "Нет открытых документов — Файл > Новый или Ctrl+V",
        "status.zoom": "Масштаб: {pct}%",
        "status.zoom_empty": "Масштаб: --",
        "status.cursor": "X: {x}, Y: {y}",
        "status.doc_size": "{w} × {h} пикс.",
        "status.no_doc": "Нет документа",

        # New Canvas Dialog
        "new_canvas.title": "Новый холст",
        "new_canvas.width": "Ширина (пикс.):",
        "new_canvas.height": "Высота (пикс.):",
        "new_canvas.background": "Фон:",
        "new_canvas.bg_white": "Белый",
        "new_canvas.bg_transparent": "Прозрачный",
        "new_canvas.bg_black": "Чёрный",

        # ToolOptionsBar
        "opts.size": "Размер:",
        "opts.px_suffix": " пикс.",
        "opts.pt_suffix": " пт",
        "opts.hardness": "Жёсткость:",
        "opts.opacity": "Непрозр.:",
        "opts.flow": "Нажим:",
        "opts.paint_mask": "Рисовать на маске",
        "opts.mode": "Режим:",
        "opts.blur": "Размытие",
        "opts.sharpen": "Резкость",
        "opts.kernel": "Ядро:",
        "opts.strength": "Сила:",
        "opts.tolerance": "Допуск:",
        "opts.contiguous": "Смежные пиксели",
        "opts.clear_selection": "Снять выделение",
        "opts.font": "Шрифт:",
        "opts.bold": "Жирный",
        "opts.text": "Текст:",
        "opts.center_h": "По гориз.",
        "opts.center_v": "По верт.",
        "opts.center_both": "По центру",
        "opts.move_hint": "Магнитная привязка активна (удерживайте Shift для свободного перемещения)",
        "opts.fit_view": "По экрану",
        "opts.hand_hint": "Перетащите холст (Пробел+ЛКМ или СКМ из любого инструмента)",
        "opts.selection_hint": "Выделите область (Shift: добавить, Alt: вычесть, Клик/Esc: сбросить)",
        "opts.crop_ratio": "Пропорции:",
        "opts.crop_free": "Свободный",
        "opts.crop_swap_tip": "Поменять ориентацию сторон (⇄)",
        "opts.crop_hint": "Выделите рамку (Shift: временно без ограничений) — Enter: применить, Esc: отмена",
        "opts.transform_hint": "Тяните маркеры для масштаба/поворота (Shift: пропорции) — Enter: применить",
        "opts.apply_enter": "Применить (Enter)",
        "opts.cancel_esc": "Отмена (Esc)",
        "opts.gradient_hint": "Проведите линию на холсте для линейного градиента (основной → фоновый)",
        "opts.eyedropper_hint": "Нажмите на холст для взятия цвета (также Alt+Клик в Кисти/Заливке)",

        # ColorPanel
        "color.fg_tip": "Основной цвет (нажмите для выбора)",
        "color.bg_tip": "Фоновый цвет (нажмите для выбора)",
        "color.swap": "Поменять",
        "color.swap_tip": "Поменять основной и фоновый цвета местами (X)",
        "color.reset": "Ч / Б",
        "color.reset_tip": "Сбросить цвета: белый и чёрный (D)",
        "color.swatch_tip": "RGB({r}, {g}, {b}) — назначить основным",
        "color.pick_fg": "Выбор основного цвета",
        "color.pick_bg": "Выбор фонового цвета",
        "color.readout": "Осн.: #{fg_hex} ({fa})   Фон: #{bg_hex}",

        # LayersPanel
        "blend.normal": "Обычный",
        "blend.multiply": "Умножение",
        "blend.screen": "Экран",
        "blend.overlay": "Перекрытие",
        "blend.add": "Сложение (Add)",
        "layers.opacity": "Непрозр.:",
        "layers.vis_tip": "Включить/выключить видимость слоя",
        "layers.clip_tip": "Обтравочная маска к нижнему слою",
        "layers.mask_tip": "К слою прикреплена маска",
        "layers.locked_tip": "Слой заблокирован",
        "layers.btn_vis": "👁 Вид",
        "layers.btn_vis_tip": "Переключить видимость активного слоя",
        "layers.btn_lock": "🔒 Блок",
        "layers.btn_lock_tip": "Заблокировать / разблокировать активный слой",
        "layers.btn_clip": "↳ Обтр.",
        "layers.btn_clip_tip": "Переключить обтравочную маску",
        "layers.btn_mask_add": "+ Маска",
        "layers.btn_mask_del": "- Маска",
        "layers.btn_mask_tip": "Добавить или удалить маску слоя",
        "layers.btn_mask_toggle": "Маска вкл",
        "layers.btn_mask_toggle_tip": "Включить / отключить действие маски",
        "layers.chk_edit_mask": "Редактировать маску слоя",
        "layers.btn_add": "+ Слой",
        "layers.btn_add_tip": "Создать новый слой",
        "layers.btn_dup": "Дубл.",
        "layers.btn_dup_tip": "Дублировать активный слой",
        "layers.btn_up_tip": "Поднять слой выше",
        "layers.btn_down_tip": "Опустить слой ниже",
        "layers.btn_merge": "Слить",
        "layers.btn_merge_tip": "Объединить с нижним слоем",
        "layers.btn_del": "Удал.",
        "layers.btn_del_tip": "Удалить активный слой",
        "layers.rename_title": "Переименовать слой",
        "layers.rename_prompt": "Название слоя:",

        # Dialogs shared
        "dlg.open_title": "Открыть изображение",
        "dlg.save_title": "Сохранить изображение",
        "dlg.select_folder": "Выберите папку для сохранения",
        "dlg.close": "Закрыть",
        "dlg.cancel": "Отмена",
        "dlg.apply": "Применить",
        "dlg.browse": "Обзор...",
        "dlg.source": "Источник:",
        "dlg.source_canvas": "Сведённый холст",
        "dlg.source_layer": "Активный слой",
        "dlg.export_pngs": "Экспорт в PNG...",
        "dlg.export_zip": "Сохранить как ZIP-архив...",
        "dlg.open_tabs": "Открыть во вкладках",
        "splitter.selected_count": "Выбрано: {selected} из {total} (клик по рамке — исключить)",
        "splitter.zip_save_title": "Сохранить ZIP-архив",
        "splitter.zip_exported_msg": "Сохранено фрагментов в ZIP ({count}):\n{path}",

        # Canvas Context Menu & Selection Actions
        "ctx.deselect": "Снять выделение (Ctrl+D)",
        "ctx.invert_sel": "Инвертировать выделение (Ctrl+Shift+I)",
        "ctx.layer_via_copy": "Скопировать на новый слой (Ctrl+J)",
        "ctx.layer_via_cut": "Вырезать на новый слой (Ctrl+Shift+J)",
        "ctx.free_transform": "Свободное трансформирование (Ctrl+T)",
        "ctx.fill_fg": "Залить основным цветом",
        "ctx.clear_sel_contents": "Удалить содержимое (Del)",
        "ctx.crop_to_sel": "Кадрировать по выделению",
        "ctx.dup_layer": "Дублировать слой",
        "ctx.center_canvas": "Выровнять по центру холста",
        "ctx.paste": "Вставить из буфера (Ctrl+V)",

        # QuickExportDialog
        "export.title": "Быстрый экспорт",
        "export.est_size": "Расчётный вес: {size}",
        "export.format": "Формат:",
        "export.quality": "Качество:",
        "export.scale": "Масштаб:",
        "export.btn": "Экспортировать...",
        "export.save_title": "Экспорт изображения",

        # ChromaKeyDialog
        "chroma.title": "Хромакей (удаление фона по цвету)",
        "chroma.key_color": "Ключевой цвет",
        "chroma.eyedropper_corner": "Пипетка (угол)",
        "chroma.pick_color_title": "Выбор ключевого цвета",
        "chroma.tolerance": "Допуск:",
        "chroma.softness": "Мягкость краёв:",
        "chroma.despill": "Устранение рефлекса:",
        "chroma.locked_warn": "Активный слой отсутствует или заблокирован.",

        # GridSplitterDialog
        "grid.title": "Нарезка по сетке",
        "grid.mode_box": "Режим нарезки",
        "grid.rows_cols": "Строки × Столбцы",
        "grid.fixed_tile": "Фиксированный размер тайла (пикс.)",
        "grid.rows": "Строк:",
        "grid.cols": "Столбцов:",
        "grid.tile_w": "Ширина:",
        "grid.tile_h": "Высота:",
        "grid.prefix": "Префикс имени:",
        "grid.summary": "Фрагментов: {total} ({w}×{h} пикс.)",
        "grid.exported_msg": "Экспортировано фрагментов: {count}\n{folder}",

        # SmartAlphaSplitterDialog
        "alpha_split.title": "Умная нарезка по альфа-каналу",
        "alpha_split.threshold": "Порог альфы:",
        "alpha_split.min_area": "Мин. площадь (пикс.):",
        "alpha_split.padding": "Отступ (пикс.):",
        "alpha_split.prefix": "Префикс:",
        "alpha_split.auto_preview": "Показывать рамки на холсте автоматически",
        "alpha_split.detected": "Найдено объектов: {count}",
        "alpha_split.preview_btn": "Показать на холсте",
        "alpha_split.exported_msg": "Экспортировано объектов: {count}\n{folder}",

        # BatchProcessorDialog
        "batch.title": "Пакетная обработка изображений",
        "batch.in_dir": "Входная папка:",
        "batch.out_dir": "Папка вывода:",
        "batch.out_settings": "Формат вывода:",
        "batch.threads": "Потоки:",
        "batch.ops_box": "Операции конвейера",
        "batch.op_ai_bg": "ИИ-удаление фона (u2netp.onnx)",
        "batch.op_chroma": "Хромакей",
        "batch.op_tol": "Допуск:",
        "batch.op_trim": "Обрезать прозрачные края",
        "batch.op_resize": "Изменить размер",
        "batch.w": "Ш:",
        "batch.h": "В:",
        "batch.ready": "Готово к запуску",
        "batch.run": "Запустить",
        "batch.sel_in": "Выберите входную папку",
        "batch.sel_out": "Выберите выходную папку",
        "batch.warn_dirs": "Укажите входную и выходную папки.",
        "batch.warn_missing_in": "Входная папка не существует:\n{path}",
        "batch.starting": "Запуск пакетной обработки...",
        "batch.done": "Готово: {ok}/{total} успешно, ошибок: {fail}",
        "batch.error": "Ошибка пакетной обработки",

        # HotkeyEditorDialog
        "hotkeys.title": "Настройка горячих клавиш",
        "hotkeys.col_action": "Действие",
        "hotkeys.col_shortcut": "Сочетание",
        "hotkeys.reset": "Сбросить по умолчанию",
        "hk.move": "Инструмент: Перемещение",
        "hk.hand": "Инструмент: Рука",
        "hk.brush": "Инструмент: Кисть",
        "hk.eraser": "Инструмент: Ластик",
        "hk.smudge": "Инструмент: Палец",
        "hk.blur_sharpen": "Инструмент: Размытие / Резкость",
        "hk.marquee": "Инструмент: Прямоугольная область",
        "hk.lasso": "Инструмент: Лассо",
        "hk.magic_wand": "Инструмент: Волшебная палочка",
        "hk.crop": "Инструмент: Кадрирование",
        "hk.transform": "Инструмент: Свободное трансформирование",
        "hk.eyedropper": "Инструмент: Пипетка",
        "hk.bucket_fill": "Инструмент: Заливка",
        "hk.gradient": "Инструмент: Градиент",
        "hk.text": "Инструмент: Текст",
        "hk.new_tab": "Файл: Новый документ",
        "hk.open_file": "Файл: Открыть изображение",
        "hk.save_file": "Файл: Сохранить",
        "hk.quick_export": "Файл: Быстрый экспорт",
        "hk.undo": "Правка: Отменить",
        "hk.redo": "Правка: Повторить",
        "hk.paste_clipboard": "Правка: Вставить / Из буфера",
        "hk.zen_mode": "Вид: Режим чистого холста (Zen)",
        "hk.center_both": "Слой: Выровнять по центру холста",
    },
    "en": {
        "app.title": "LitePhotoEditor",
        "menu.file": "&File",
        "menu.edit": "&Edit",
        "menu.image": "&Image",
        "menu.layer": "&Layer",
        "menu.tools": "&Tools",
        "menu.view": "&View",
        "menu.language": "&Language",
        "lang.ru": "Русский (RU)",
        "lang.en": "English (EN)",

        "action.new_tab": "&New Tab",
        "action.new_custom": "New &Custom Canvas...",
        "action.open_file": "&Open...",
        "action.save_file": "&Save",
        "action.quick_export": "Quick &Export...",
        "action.batch": "&Batch Processor...",
        "action.close_tab": "&Close Tab",
        "action.exit": "E&xit",

        "action.undo": "&Undo",
        "action.redo": "&Redo",
        "action.copy": "&Copy Selection",
        "action.paste": "&Paste / New Tab from Clipboard",
        "action.deselect": "&Deselect",
        "action.hotkeys": "&Keyboard Shortcuts...",

        "action.trim": "&Trim Transparent Pixels",
        "action.ai_bg": "&AI Remove Background",
        "action.chroma_key": "&Chroma Key...",
        "action.grid_split": "&Grid Splitter...",
        "action.alpha_split": "Smart &Alpha Splitter...",

        "action.add_layer": "&Add Layer",
        "action.dup_layer": "&Duplicate Layer",
        "action.del_layer": "De&lete Layer",
        "action.merge_down": "&Merge Down",
        "action.toggle_mask_add_del": "Add / Delete Layer &Mask",
        "action.toggle_clip": "Toggle &Clipping Mask",
        "action.center_h": "Center &Horizontal",
        "action.center_v": "Center &Vertical",
        "action.center_both": "Center &Both",

        "action.swap_colors": "Swap FG/BG Colors",
        "action.reset_colors": "Reset B/W Colors",

        "action.zoom_in": "Zoom &In",
        "action.zoom_out": "Zoom &Out",
        "action.fit_screen": "&Fit to Screen",
        "action.actual_size": "&Actual Size (100%)",
        "action.zen_mode": "&Zen Mode",

        "tool.short.move": "Move",
        "tool.short.hand": "Hand",
        "tool.short.marquee": "Marquee",
        "tool.short.lasso": "Lasso",
        "tool.short.magic_wand": "Wand",
        "tool.short.crop": "Crop",
        "tool.short.transform": "Transform",
        "tool.short.brush": "Brush",
        "tool.short.eraser": "Eraser",
        "tool.short.smudge": "Smudge",
        "tool.short.blur_sharpen": "Blur/Sharp",
        "tool.short.bucket_fill": "Fill",
        "tool.short.gradient": "Gradient",
        "tool.short.eyedropper": "Picker",
        "tool.short.text": "Text",

        "tool.name.move": "Move",
        "tool.name.hand": "Hand",
        "tool.name.marquee": "Rect Marquee",
        "tool.name.lasso": "Lasso",
        "tool.name.magic_wand": "Magic Wand",
        "tool.name.crop": "Crop",
        "tool.name.transform": "Free Transform",
        "tool.name.brush": "Brush",
        "tool.name.eraser": "Eraser",
        "tool.name.smudge": "Smudge",
        "tool.name.blur_sharpen": "Blur / Sharpen",
        "tool.name.bucket_fill": "Bucket Fill",
        "tool.name.gradient": "Gradient",
        "tool.name.eyedropper": "Eyedropper",
        "tool.name.text": "Text",

        "dock.colors": "Colors & Swatches",
        "dock.layers": "Layers",
        "dock.tools": "Tools",
        "tab.new_btn_tip": "New Document Tab (Ctrl+N)",
        "doc.untitled": "Untitled",
        "doc.clipboard_prefix": "Clipboard",
        "doc.layer_pasted": "Pasted Layer",
        "doc.background_layer": "Background",

        "canvas.empty_hint": "No Document Open — File > New or Ctrl+V",
        "status.zoom": "Zoom: {pct}%",
        "status.zoom_empty": "Zoom: --",
        "status.cursor": "X: {x}, Y: {y}",
        "status.doc_size": "{w} x {h} px",
        "status.no_doc": "No document",

        "new_canvas.title": "New Canvas",
        "new_canvas.width": "Width (px):",
        "new_canvas.height": "Height (px):",
        "new_canvas.background": "Background:",
        "new_canvas.bg_white": "White",
        "new_canvas.bg_transparent": "Transparent",
        "new_canvas.bg_black": "Black",

        "opts.size": "Size:",
        "opts.px_suffix": " px",
        "opts.pt_suffix": " pt",
        "opts.hardness": "Hardness:",
        "opts.opacity": "Opacity:",
        "opts.flow": "Flow:",
        "opts.paint_mask": "Paint Mask",
        "opts.mode": "Mode:",
        "opts.blur": "Blur",
        "opts.sharpen": "Sharpen",
        "opts.kernel": "Kernel:",
        "opts.strength": "Strength:",
        "opts.tolerance": "Tolerance:",
        "opts.contiguous": "Contiguous",
        "opts.clear_selection": "Clear Selection",
        "opts.font": "Font:",
        "opts.bold": "Bold",
        "opts.text": "Text:",
        "opts.center_h": "Center H",
        "opts.center_v": "Center V",
        "opts.center_both": "Center Both",
        "opts.move_hint": "Magnetic snap enabled (hold Shift for free move)",
        "opts.fit_view": "Fit to View",
        "opts.hand_hint": "Drag to pan viewport (Space+Drag or Middle-Drag from any tool)",
        "opts.selection_hint": "Drag to select (Shift: Add, Alt: Subtract, Click/Esc: Deselect)",
        "opts.crop_ratio": "Aspect:",
        "opts.crop_free": "Free",
        "opts.crop_swap_tip": "Swap Orientation (⇄)",
        "opts.crop_hint": "Drag crop box (Shift: free ratio) — Enter: apply, Esc: cancel",
        "opts.transform_hint": "Drag handles to scale/rotate (Shift: lock aspect) — Enter to commit",
        "opts.apply_enter": "Apply (Enter)",
        "opts.cancel_esc": "Cancel (Esc)",
        "opts.gradient_hint": "Drag on canvas to draw a linear gradient from FG to BG color",
        "opts.eyedropper_hint": "Click on canvas to sample Foreground color (also Alt+Click with Brush/Fill)",

        "color.fg_tip": "Foreground Color (Click to pick)",
        "color.bg_tip": "Background Color (Click to pick)",
        "color.swap": "Swap FG/BG",
        "color.swap_tip": "Swap Foreground and Background Colors (X)",
        "color.reset": "Reset B/W",
        "color.reset_tip": "Reset Foreground=White / Background=Black (D)",
        "color.swatch_tip": "RGB({r}, {g}, {b}) — Click to set FG",
        "color.pick_fg": "Select Foreground Color",
        "color.pick_bg": "Select Background Color",
        "color.readout": "FG: #{fg_hex} ({fa})   BG: #{bg_hex}",

        "blend.normal": "Normal",
        "blend.multiply": "Multiply",
        "blend.screen": "Screen",
        "blend.overlay": "Overlay",
        "blend.add": "Add",
        "layers.opacity": "Opacity:",
        "layers.vis_tip": "Toggle Layer Visibility",
        "layers.clip_tip": "Clipped to base layer below",
        "layers.mask_tip": "Layer Mask attached",
        "layers.locked_tip": "Layer Locked",
        "layers.btn_vis": "👁 Vis",
        "layers.btn_vis_tip": "Toggle Active Layer Visibility",
        "layers.btn_lock": "🔒 Lock",
        "layers.btn_lock_tip": "Toggle Active Layer Lock",
        "layers.btn_clip": "↳ Clip",
        "layers.btn_clip_tip": "Toggle Clipping Mask to Layer Below",
        "layers.btn_mask_add": "+ Mask",
        "layers.btn_mask_del": "- Mask",
        "layers.btn_mask_tip": "Add or Delete Layer Mask",
        "layers.btn_mask_toggle": "M On/Off",
        "layers.btn_mask_toggle_tip": "Enable / Disable Layer Mask",
        "layers.chk_edit_mask": "Paint on Layer Mask",
        "layers.btn_add": "+ Add",
        "layers.btn_add_tip": "Add New Layer",
        "layers.btn_dup": "Dup",
        "layers.btn_dup_tip": "Duplicate Active Layer",
        "layers.btn_up_tip": "Move Layer Up",
        "layers.btn_down_tip": "Move Layer Down",
        "layers.btn_merge": "Merge",
        "layers.btn_merge_tip": "Merge Down",
        "layers.btn_del": "Del",
        "layers.btn_del_tip": "Delete Active Layer",
        "layers.rename_title": "Rename Layer",
        "layers.rename_prompt": "Layer Name:",

        "dlg.open_title": "Open Image",
        "dlg.save_title": "Save Image",
        "dlg.select_folder": "Select Output Folder",
        "dlg.close": "Close",
        "dlg.cancel": "Cancel",
        "dlg.apply": "Apply",
        "dlg.browse": "Browse...",
        "dlg.source": "Source:",
        "dlg.source_canvas": "Composited Canvas",
        "dlg.source_layer": "Active Layer",
        "dlg.export_pngs": "Export PNGs...",
        "dlg.export_zip": "Export to .zip...",
        "dlg.open_tabs": "Open in Tabs",
        "splitter.selected_count": "Selected: {selected} of {total} (click a box to exclude)",
        "splitter.zip_save_title": "Save ZIP Archive",
        "splitter.zip_exported_msg": "Saved {count} slice(s) to ZIP:\n{path}",

        "ctx.deselect": "Deselect (Ctrl+D)",
        "ctx.invert_sel": "Invert Selection (Ctrl+Shift+I)",
        "ctx.layer_via_copy": "Layer via Copy (Ctrl+J)",
        "ctx.layer_via_cut": "Layer via Cut (Ctrl+Shift+J)",
        "ctx.free_transform": "Free Transform (Ctrl+T)",
        "ctx.fill_fg": "Fill with Foreground Color",
        "ctx.clear_sel_contents": "Clear Selection Contents (Del)",
        "ctx.crop_to_sel": "Crop to Selection",
        "ctx.dup_layer": "Duplicate Layer",
        "ctx.center_canvas": "Center on Canvas",
        "ctx.paste": "Paste (Ctrl+V)",

        "export.title": "Quick Export",
        "export.est_size": "Estimated Size: {size}",
        "export.format": "Format:",
        "export.quality": "Quality:",
        "export.scale": "Scale:",
        "export.btn": "Export...",
        "export.save_title": "Export Image",

        "chroma.title": "Chroma Key (Green / Solid Screen)",
        "chroma.key_color": "Key Color",
        "chroma.eyedropper_corner": "Eyedropper (Corner)",
        "chroma.pick_color_title": "Pick Chroma Key Color",
        "chroma.tolerance": "Tolerance:",
        "chroma.softness": "Edge Softness:",
        "chroma.despill": "Despill:",
        "chroma.locked_warn": "Active layer is missing or locked; chroma key cannot edit pixels.",

        "grid.title": "Grid Image Splitter",
        "grid.mode_box": "Split Mode",
        "grid.rows_cols": "Rows x Columns",
        "grid.fixed_tile": "Fixed Tile Size (px)",
        "grid.rows": "Rows:",
        "grid.cols": "Cols:",
        "grid.tile_w": "Tile W:",
        "grid.tile_h": "Tile H:",
        "grid.prefix": "Filename Prefix:",
        "grid.summary": "{total} slice(s) from {w}x{h} px",
        "grid.exported_msg": "Exported {count} slice(s) to:\n{folder}",

        "alpha_split.title": "Smart Alpha Splitter",
        "alpha_split.threshold": "Alpha Threshold:",
        "alpha_split.min_area": "Min Area (px):",
        "alpha_split.padding": "Padding (px):",
        "alpha_split.prefix": "Prefix:",
        "alpha_split.auto_preview": "Auto Preview Boxes on Canvas",
        "alpha_split.detected": "Detected {count} sprite(s)",
        "alpha_split.preview_btn": "Preview on Canvas",
        "alpha_split.exported_msg": "Exported {count} sprite(s) to:\n{folder}",

        "batch.title": "Batch Image Processor",
        "batch.in_dir": "Input Folder:",
        "batch.out_dir": "Output Folder:",
        "batch.out_settings": "Output Settings:",
        "batch.threads": "Threads:",
        "batch.ops_box": "Pipeline Operations",
        "batch.op_ai_bg": "AI Remove Background (u2netp.onnx)",
        "batch.op_chroma": "Chroma Key",
        "batch.op_tol": "Tol:",
        "batch.op_trim": "Trim Transparent Pixels",
        "batch.op_resize": "Resize",
        "batch.w": "W:",
        "batch.h": "H:",
        "batch.ready": "Ready",
        "batch.run": "Run Batch",
        "batch.sel_in": "Select Input Folder",
        "batch.sel_out": "Select Output Folder",
        "batch.warn_dirs": "Please specify both Input and Output folders.",
        "batch.warn_missing_in": "Input folder does not exist:\n{path}",
        "batch.starting": "Starting batch...",
        "batch.done": "Done: {ok}/{total} succeeded, {fail} failed",
        "batch.error": "Batch Error",

        "hotkeys.title": "Keyboard Shortcuts",
        "hotkeys.col_action": "Action",
        "hotkeys.col_shortcut": "Shortcut",
        "hotkeys.reset": "Reset Defaults",
        "hk.move": "Tool: Move",
        "hk.hand": "Tool: Hand",
        "hk.brush": "Tool: Brush",
        "hk.eraser": "Tool: Eraser",
        "hk.smudge": "Tool: Smudge",
        "hk.blur_sharpen": "Tool: Blur / Sharpen",
        "hk.marquee": "Tool: Rect Marquee",
        "hk.lasso": "Tool: Lasso",
        "hk.magic_wand": "Tool: Magic Wand",
        "hk.crop": "Tool: Crop",
        "hk.transform": "Tool: Free Transform",
        "hk.eyedropper": "Tool: Eyedropper",
        "hk.bucket_fill": "Tool: Bucket Fill",
        "hk.gradient": "Tool: Linear Gradient",
        "hk.text": "Tool: Text",
        "hk.new_tab": "File: New Document Tab",
        "hk.open_file": "File: Open Image",
        "hk.save_file": "File: Save",
        "hk.quick_export": "File: Quick Export",
        "hk.undo": "Edit: Undo",
        "hk.redo": "Edit: Redo",
        "hk.paste_clipboard": "Edit: Paste / Clipboard Tab",
        "hk.zen_mode": "View: Toggle Zen Mode",
        "hk.center_both": "Layer: Center on Canvas",
    },
}


def resolve_ui_settings_path(for_write: bool = False) -> Path:
    """Resolve app/config/ui_settings.json path across workspace and PyInstaller builds."""
    rel = Path("app") / "config" / "ui_settings.json"
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        exe_path = Path(sys.executable).resolve().parent / rel
        if for_write or exe_path.exists():
            return exe_path
        return Path(sys._MEIPASS) / rel  # type: ignore[attr-defined]
    return Path(__file__).resolve().parents[2] / rel


def _load_saved_language() -> str:
    cfg = resolve_ui_settings_path(for_write=False)
    if cfg.is_file():
        try:
            data = json.loads(cfg.read_text(encoding="utf-8"))
            lang = str(data.get("language", DEFAULT_LANGUAGE)).lower()
            if lang in SUPPORTED_LANGUAGES:
                return lang
        except Exception:
            pass
    return DEFAULT_LANGUAGE


_CURRENT_LANG: str = _load_saved_language()
_LISTENERS: list[Callable[[str], None]] = []


def get_language() -> str:
    return _CURRENT_LANG


def set_language(lang: str, persist: bool = True) -> str:
    global _CURRENT_LANG
    norm = str(lang).strip().lower()
    if norm not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language: {lang!r}")
    _CURRENT_LANG = norm
    if persist:
        cfg = resolve_ui_settings_path(for_write=True)
        try:
            cfg.parent.mkdir(parents=True, exist_ok=True)
            data: dict = {}
            if cfg.is_file():
                try:
                    data = json.loads(cfg.read_text(encoding="utf-8"))
                    if not isinstance(data, dict):
                        data = {}
                except Exception:
                    data = {}
            data["language"] = norm
            cfg.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        except OSError:
            pass
    for cb in list(_LISTENERS):
        try:
            cb(norm)
        except Exception:
            pass
    return norm


def register_language_listener(callback: Callable[[str], None]) -> None:
    if callback not in _LISTENERS:
        _LISTENERS.append(callback)


def tr(key: str, **kwargs) -> str:
    lang_map = _TRANSLATIONS.get(_CURRENT_LANG, _TRANSLATIONS[DEFAULT_LANGUAGE])
    template = lang_map.get(key)
    if template is None:
        template = _TRANSLATIONS["en"].get(key, key)
    if kwargs:
        try:
            return template.format(**kwargs)
        except Exception:
            return template
    return template
