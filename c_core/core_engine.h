#ifndef CORE_ENGINE_H
#define CORE_ENGINE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

/* Blend modes supported by composite_layer */
#define BLEND_NORMAL   0
#define BLEND_MULTIPLY 1
#define BLEND_SCREEN   2
#define BLEND_OVERLAY  3
#define BLEND_ADD      4

/* Blur / Sharpen brush modes */
#define FILTER_MODE_BLUR    0
#define FILTER_MODE_SHARPEN 1

typedef struct {
    int x;
    int y;
    int width;
    int height;
    int area;
    int label;
} ComponentBox;

EXPORT void clear_buffer_rgba(
    uint8_t* dst,
    int width,
    int height,
    uint8_t r,
    uint8_t g,
    uint8_t b,
    uint8_t a
);

EXPORT void composite_layer(
    uint8_t* dst,
    int dst_w,
    int dst_h,
    const uint8_t* src,
    int src_w,
    int src_h,
    const uint8_t* mask,
    const uint8_t* clip_alpha,
    int offset_x,
    int offset_y,
    float opacity,
    int blend_mode,
    int roi_x,
    int roi_y,
    int roi_w,
    int roi_h
);

EXPORT void extract_alpha(
    const uint8_t* src_rgba,
    uint8_t* dst_alpha,
    int width,
    int height
);

EXPORT void brush_dab_rgba(
    uint8_t* dst,
    int width,
    int height,
    const uint8_t* selection_mask,
    float cx,
    float cy,
    float radius,
    float hardness,
    uint8_t r,
    uint8_t g,
    uint8_t b,
    uint8_t a,
    float opacity,
    float flow,
    int is_eraser,
    int* out_dirty_rect
);

EXPORT void brush_stroke_rgba(
    uint8_t* dst,
    int width,
    int height,
    const uint8_t* selection_mask,
    float x0,
    float y0,
    float x1,
    float y1,
    float radius,
    float hardness,
    float spacing,
    uint8_t r,
    uint8_t g,
    uint8_t b,
    uint8_t a,
    float opacity,
    float flow,
    int is_eraser,
    int* out_dirty_rect
);

EXPORT void brush_stroke_mask(
    uint8_t* mask,
    int width,
    int height,
    const uint8_t* selection_mask,
    float x0,
    float y0,
    float x1,
    float y1,
    float radius,
    float hardness,
    float spacing,
    uint8_t value,
    float opacity,
    float flow,
    int* out_dirty_rect
);

EXPORT void smudge_stroke_rgba(
    uint8_t* dst,
    int width,
    int height,
    const uint8_t* selection_mask,
    float x0,
    float y0,
    float x1,
    float y1,
    float radius,
    float hardness,
    float strength,
    int* out_dirty_rect
);

EXPORT void blur_sharpen_dab_rgba(
    uint8_t* dst,
    int width,
    int height,
    const uint8_t* selection_mask,
    float cx,
    float cy,
    float radius,
    float hardness,
    float strength,
    int mode,
    int kernel_size,
    int* out_dirty_rect
);

EXPORT void gaussian_blur_rgba(
    const uint8_t* src,
    uint8_t* dst,
    int width,
    int height,
    int radius
);

EXPORT void chroma_key_rgba(
    uint8_t* rgba,
    int width,
    int height,
    uint8_t key_r,
    uint8_t key_g,
    uint8_t key_b,
    float tolerance,
    float softness,
    float despill
);

EXPORT int flood_fill_rgba(
    uint8_t* rgba,
    int width,
    int height,
    const uint8_t* selection_mask,
    int seed_x,
    int seed_y,
    uint8_t fill_r,
    uint8_t fill_g,
    uint8_t fill_b,
    uint8_t fill_a,
    int tolerance,
    int contiguous,
    int* out_dirty_rect
);

EXPORT int magic_wand_select(
    const uint8_t* rgba,
    int width,
    int height,
    int seed_x,
    int seed_y,
    int tolerance,
    int contiguous,
    uint8_t* out_mask,
    int* out_dirty_rect
);

EXPORT int connected_components_alpha(
    const uint8_t* rgba,
    int width,
    int height,
    uint8_t alpha_threshold,
    int min_area,
    int padding,
    int32_t* out_labels,
    ComponentBox* out_boxes,
    int max_boxes
);

EXPORT int find_alpha_bounds(
    const uint8_t* rgba,
    int width,
    int height,
    uint8_t alpha_threshold,
    int* out_rect
);

EXPORT void bilinear_resize_rgba(
    const uint8_t* src,
    int src_w,
    int src_h,
    uint8_t* dst,
    int dst_w,
    int dst_h
);

#ifdef __cplusplus
}
#endif

#endif /* CORE_ENGINE_H */
