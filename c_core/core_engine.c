#include "core_engine.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static inline int clamp_int(int v, int lo, int hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static inline float clamp_float(float v, float lo, float hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static inline uint8_t float_to_u8(float v) {
    int iv = (int)(v + 0.5f);
    if (iv <= 0) return 0;
    if (iv >= 255) return 255;
    return (uint8_t)iv;
}

static inline float calc_radial_falloff(float dist, float radius, float hardness) {
    if (radius <= 0.0f || dist >= radius) {
        return 0.0f;
    }
    float t = dist / radius;
    float h = clamp_float(hardness, 0.0f, 1.0f);
    if (h >= 0.999f || t <= h) {
        return 1.0f;
    }
    float u = (t - h) / (1.0f - h);
    return 0.5f * (1.0f + cosf((float)M_PI * u));
}

static inline float apply_blend_channel(float d, float s, int blend_mode) {
    switch (blend_mode) {
        case BLEND_MULTIPLY:
            return d * s;
        case BLEND_SCREEN:
            return 1.0f - (1.0f - d) * (1.0f - s);
        case BLEND_OVERLAY:
            return (d < 0.5f)
                ? (2.0f * d * s)
                : (1.0f - 2.0f * (1.0f - d) * (1.0f - s));
        case BLEND_ADD: {
            float sum = d + s;
            return sum > 1.0f ? 1.0f : sum;
        }
        case BLEND_NORMAL:
        default:
            return s;
    }
}

static void write_dirty_rect(int* out_dirty_rect, int min_x, int min_y, int max_x, int max_y) {
    if (!out_dirty_rect) return;
    if (max_x < min_x || max_y < min_y) {
        out_dirty_rect[0] = 0;
        out_dirty_rect[1] = 0;
        out_dirty_rect[2] = 0;
        out_dirty_rect[3] = 0;
    } else {
        out_dirty_rect[0] = min_x;
        out_dirty_rect[1] = min_y;
        out_dirty_rect[2] = max_x - min_x + 1;
        out_dirty_rect[3] = max_y - min_y + 1;
    }
}

EXPORT void clear_buffer_rgba(
    uint8_t* dst,
    int width,
    int height,
    uint8_t r,
    uint8_t g,
    uint8_t b,
    uint8_t a
) {
    if (!dst || width <= 0 || height <= 0) return;
    const int64_t total = (int64_t)width * (int64_t)height;
    int64_t i;
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(total > 32768)
    #endif
    for (i = 0; i < total; ++i) {
        const int64_t idx = i * 4;
        dst[idx + 0] = r;
        dst[idx + 1] = g;
        dst[idx + 2] = b;
        dst[idx + 3] = a;
    }
}

EXPORT void extract_alpha(
    const uint8_t* src_rgba,
    uint8_t* dst_alpha,
    int width,
    int height
) {
    if (!src_rgba || !dst_alpha || width <= 0 || height <= 0) return;
    const int64_t total = (int64_t)width * (int64_t)height;
    int64_t i;
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(total > 32768)
    #endif
    for (i = 0; i < total; ++i) {
        dst_alpha[i] = src_rgba[i * 4 + 3];
    }
}

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
) {
    if (!dst || !src || dst_w <= 0 || dst_h <= 0 || src_w <= 0 || src_h <= 0) {
        return;
    }
    const float op = clamp_float(opacity, 0.0f, 1.0f);
    if (op <= 0.0f) return;

    int rx0 = 0;
    int ry0 = 0;
    int rx1 = dst_w;
    int ry1 = dst_h;
    if (roi_w > 0 && roi_h > 0) {
        rx0 = clamp_int(roi_x, 0, dst_w);
        ry0 = clamp_int(roi_y, 0, dst_h);
        rx1 = clamp_int(roi_x + roi_w, 0, dst_w);
        ry1 = clamp_int(roi_y + roi_h, 0, dst_h);
    }

    const int x_start = clamp_int(offset_x, rx0, rx1);
    const int y_start = clamp_int(offset_y, ry0, ry1);
    const int x_end = clamp_int(offset_x + src_w, rx0, rx1);
    const int y_end = clamp_int(offset_y + src_h, ry0, ry1);

    if (x_start >= x_end || y_start >= y_end) return;

    const int64_t area = (int64_t)(x_end - x_start) * (int64_t)(y_end - y_start);
    const float inv255 = 1.0f / 255.0f;

    int dy;
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(area > 16384)
    #endif
    for (dy = y_start; dy < y_end; ++dy) {
        const int sy = dy - offset_y;
        const int dst_row_offset = dy * dst_w;
        const int src_row_offset = sy * src_w;

        for (int dx = x_start; dx < x_end; ++dx) {
            const int sx = dx - offset_x;
            const int src_px = src_row_offset + sx;
            const int dst_px = dst_row_offset + dx;
            const int src_idx = src_px * 4;
            const int dst_idx = dst_px * 4;

            float sa = (float)src[src_idx + 3] * inv255 * op;
            if (mask) {
                sa *= (float)mask[src_px] * inv255;
            }
            if (clip_alpha) {
                sa *= (float)clip_alpha[dst_px] * inv255;
            }
            if (sa <= 0.0f) continue;

            const float sr = (float)src[src_idx + 0] * inv255;
            const float sg = (float)src[src_idx + 1] * inv255;
            const float sb = (float)src[src_idx + 2] * inv255;

            const float da = (float)dst[dst_idx + 3] * inv255;
            const float out_a = sa + da * (1.0f - sa);
            if (out_a <= 1e-6f) {
                dst[dst_idx + 0] = 0;
                dst[dst_idx + 1] = 0;
                dst[dst_idx + 2] = 0;
                dst[dst_idx + 3] = 0;
                continue;
            }

            if (da <= 1e-6f) {
                /* Destination is completely transparent: straight source color */
                dst[dst_idx + 0] = src[src_idx + 0];
                dst[dst_idx + 1] = src[src_idx + 1];
                dst[dst_idx + 2] = src[src_idx + 2];
                dst[dst_idx + 3] = float_to_u8(out_a * 255.0f);
                continue;
            }

            const float dr = (float)dst[dst_idx + 0] * inv255;
            const float dg = (float)dst[dst_idx + 1] * inv255;
            const float db = (float)dst[dst_idx + 2] * inv255;

            const float br = (blend_mode == BLEND_NORMAL) ? sr : apply_blend_channel(dr, sr, blend_mode);
            const float bg = (blend_mode == BLEND_NORMAL) ? sg : apply_blend_channel(dg, sg, blend_mode);
            const float bb = (blend_mode == BLEND_NORMAL) ? sb : apply_blend_channel(db, sb, blend_mode);

            const float k_src = (1.0f - da) * sa;
            const float k_dst = (1.0f - sa) * da;
            const float k_bld = sa * da;
            const float inv_out_a = 1.0f / out_a;

            const float out_r = (k_src * sr + k_dst * dr + k_bld * br) * inv_out_a;
            const float out_g = (k_src * sg + k_dst * dg + k_bld * bg) * inv_out_a;
            const float out_b = (k_src * sb + k_dst * db + k_bld * bb) * inv_out_a;

            dst[dst_idx + 0] = float_to_u8(out_r * 255.0f);
            dst[dst_idx + 1] = float_to_u8(out_g * 255.0f);
            dst[dst_idx + 2] = float_to_u8(out_b * 255.0f);
            dst[dst_idx + 3] = float_to_u8(out_a * 255.0f);
        }
    }
}

static void stamp_dab_rgba_internal(
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
    int* io_min_x,
    int* io_min_y,
    int* io_max_x,
    int* io_max_y
) {
    if (radius <= 0.0f) return;
    const float op = clamp_float(opacity, 0.0f, 1.0f);
    const float fl = clamp_float(flow, 0.0f, 1.0f);
    const float base_alpha = ((float)a / 255.0f) * op * fl;
    if (base_alpha <= 0.0f) return;

    const int x0 = clamp_int((int)floorf(cx - radius), 0, width - 1);
    const int y0 = clamp_int((int)floorf(cy - radius), 0, height - 1);
    const int x1 = clamp_int((int)ceilf(cx + radius), 0, width - 1);
    const int y1 = clamp_int((int)ceilf(cy + radius), 0, height - 1);

    const float inv255 = 1.0f / 255.0f;
    const float sr = (float)r * inv255;
    const float sg = (float)g * inv255;
    const float sb = (float)b * inv255;

    for (int py = y0; py <= y1; ++py) {
        const float dy = (float)py - cy;
        const int row_off = py * width;
        for (int px = x0; px <= x1; ++px) {
            const float dx = (float)px - cx;
            const float dist = sqrtf(dx * dx + dy * dy);
            const float falloff = calc_radial_falloff(dist, radius, hardness);
            if (falloff <= 0.0f) continue;

            float eff_alpha = base_alpha * falloff;
            if (selection_mask) {
                const float sel = (float)selection_mask[row_off + px] * inv255;
                if (sel <= 0.0f) continue;
                eff_alpha *= sel;
            }
            eff_alpha = clamp_float(eff_alpha, 0.0f, 1.0f);
            if (eff_alpha <= 0.0f) continue;

            const int idx = (row_off + px) * 4;
            if (is_eraser) {
                const float da = (float)dst[idx + 3] * inv255;
                const float new_da = da * (1.0f - eff_alpha);
                if (new_da <= 1e-6f) {
                    dst[idx + 0] = 0;
                    dst[idx + 1] = 0;
                    dst[idx + 2] = 0;
                    dst[idx + 3] = 0;
                } else {
                    dst[idx + 3] = float_to_u8(new_da * 255.0f);
                }
            } else {
                const float da = (float)dst[idx + 3] * inv255;
                const float out_a = eff_alpha + da * (1.0f - eff_alpha);
                if (out_a <= 1e-6f) {
                    dst[idx + 0] = 0;
                    dst[idx + 1] = 0;
                    dst[idx + 2] = 0;
                    dst[idx + 3] = 0;
                } else {
                    const float dr = (float)dst[idx + 0] * inv255;
                    const float dg = (float)dst[idx + 1] * inv255;
                    const float db = (float)dst[idx + 2] * inv255;
                    const float k_dst = da * (1.0f - eff_alpha);
                    const float inv_out = 1.0f / out_a;

                    dst[idx + 0] = float_to_u8(((sr * eff_alpha + dr * k_dst) * inv_out) * 255.0f);
                    dst[idx + 1] = float_to_u8(((sg * eff_alpha + dg * k_dst) * inv_out) * 255.0f);
                    dst[idx + 2] = float_to_u8(((sb * eff_alpha + db * k_dst) * inv_out) * 255.0f);
                    dst[idx + 3] = float_to_u8(out_a * 255.0f);
                }
            }

            if (px < *io_min_x) *io_min_x = px;
            if (py < *io_min_y) *io_min_y = py;
            if (px > *io_max_x) *io_max_x = px;
            if (py > *io_max_y) *io_max_y = py;
        }
    }
}

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
) {
    if (!dst || width <= 0 || height <= 0) {
        write_dirty_rect(out_dirty_rect, 0, 0, -1, -1);
        return;
    }
    int min_x = width, min_y = height, max_x = -1, max_y = -1;
    stamp_dab_rgba_internal(
        dst, width, height, selection_mask,
        cx, cy, radius, hardness,
        r, g, b, a, opacity, flow, is_eraser,
        &min_x, &min_y, &max_x, &max_y
    );
    write_dirty_rect(out_dirty_rect, min_x, min_y, max_x, max_y);
}

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
) {
    if (!dst || width <= 0 || height <= 0 || radius <= 0.0f) {
        write_dirty_rect(out_dirty_rect, 0, 0, -1, -1);
        return;
    }

    int min_x = width, min_y = height, max_x = -1, max_y = -1;
    const float dx = x1 - x0;
    const float dy = y1 - y0;
    const float dist = hypotf(dx, dy);
    const float step = fmaxf(1.0f, radius * clamp_float(spacing, 0.05f, 2.0f));

    if (dist < 1e-4f) {
        stamp_dab_rgba_internal(
            dst, width, height, selection_mask,
            x0, y0, radius, hardness,
            r, g, b, a, opacity, flow, is_eraser,
            &min_x, &min_y, &max_x, &max_y
        );
    } else {
        const int n_steps = (int)ceilf(dist / step);
        for (int i = 0; i <= n_steps; ++i) {
            const float t = (float)i / (float)n_steps;
            stamp_dab_rgba_internal(
                dst, width, height, selection_mask,
                x0 + dx * t, y0 + dy * t, radius, hardness,
                r, g, b, a, opacity, flow, is_eraser,
                &min_x, &min_y, &max_x, &max_y
            );
        }
    }
    write_dirty_rect(out_dirty_rect, min_x, min_y, max_x, max_y);
}

static void stamp_dab_mask_internal(
    uint8_t* mask,
    int width,
    int height,
    const uint8_t* selection_mask,
    float cx,
    float cy,
    float radius,
    float hardness,
    uint8_t value,
    float opacity,
    float flow,
    int* io_min_x,
    int* io_min_y,
    int* io_max_x,
    int* io_max_y
) {
    if (radius <= 0.0f) return;
    const float op = clamp_float(opacity, 0.0f, 1.0f);
    const float fl = clamp_float(flow, 0.0f, 1.0f);
    const float base_eff = op * fl;
    if (base_eff <= 0.0f) return;

    const int x0 = clamp_int((int)floorf(cx - radius), 0, width - 1);
    const int y0 = clamp_int((int)floorf(cy - radius), 0, height - 1);
    const int x1 = clamp_int((int)ceilf(cx + radius), 0, width - 1);
    const int y1 = clamp_int((int)ceilf(cy + radius), 0, height - 1);
    const float inv255 = 1.0f / 255.0f;
    const float target = (float)value;

    for (int py = y0; py <= y1; ++py) {
        const float dy = (float)py - cy;
        const int row_off = py * width;
        for (int px = x0; px <= x1; ++px) {
            const float dx = (float)px - cx;
            const float dist = sqrtf(dx * dx + dy * dy);
            const float falloff = calc_radial_falloff(dist, radius, hardness);
            if (falloff <= 0.0f) continue;

            float eff = base_eff * falloff;
            if (selection_mask) {
                const float sel = (float)selection_mask[row_off + px] * inv255;
                if (sel <= 0.0f) continue;
                eff *= sel;
            }
            eff = clamp_float(eff, 0.0f, 1.0f);
            if (eff <= 0.0f) continue;

            const int idx = row_off + px;
            const float cur = (float)mask[idx];
            mask[idx] = float_to_u8(cur * (1.0f - eff) + target * eff);

            if (px < *io_min_x) *io_min_x = px;
            if (py < *io_min_y) *io_min_y = py;
            if (px > *io_max_x) *io_max_x = px;
            if (py > *io_max_y) *io_max_y = py;
        }
    }
}

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
) {
    if (!mask || width <= 0 || height <= 0 || radius <= 0.0f) {
        write_dirty_rect(out_dirty_rect, 0, 0, -1, -1);
        return;
    }

    int min_x = width, min_y = height, max_x = -1, max_y = -1;
    const float dx = x1 - x0;
    const float dy = y1 - y0;
    const float dist = hypotf(dx, dy);
    const float step = fmaxf(1.0f, radius * clamp_float(spacing, 0.05f, 2.0f));

    if (dist < 1e-4f) {
        stamp_dab_mask_internal(
            mask, width, height, selection_mask,
            x0, y0, radius, hardness,
            value, opacity, flow,
            &min_x, &min_y, &max_x, &max_y
        );
    } else {
        const int n_steps = (int)ceilf(dist / step);
        for (int i = 0; i <= n_steps; ++i) {
            const float t = (float)i / (float)n_steps;
            stamp_dab_mask_internal(
                mask, width, height, selection_mask,
                x0 + dx * t, y0 + dy * t, radius, hardness,
                value, opacity, flow,
                &min_x, &min_y, &max_x, &max_y
            );
        }
    }
    write_dirty_rect(out_dirty_rect, min_x, min_y, max_x, max_y);
}

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
) {
    if (!dst || width <= 0 || height <= 0 || radius <= 0.0f) {
        write_dirty_rect(out_dirty_rect, 0, 0, -1, -1);
        return;
    }
    const float str = clamp_float(strength, 0.0f, 1.0f);
    if (str <= 0.0f) {
        write_dirty_rect(out_dirty_rect, 0, 0, -1, -1);
        return;
    }

    const int R = (int)ceilf(radius);
    const int side = 2 * R + 1;
    float* pickup = (float*)malloc((size_t)side * (size_t)side * 4 * sizeof(float));
    if (!pickup) {
        write_dirty_rect(out_dirty_rect, 0, 0, -1, -1);
        return;
    }

    const float inv255 = 1.0f / 255.0f;
    const int ix0 = (int)roundf(x0);
    const int iy0 = (int)roundf(y0);

    for (int oy = -R; oy <= R; ++oy) {
        const int sy = clamp_int(iy0 + oy, 0, height - 1);
        for (int ox = -R; ox <= R; ++ox) {
            const int sx = clamp_int(ix0 + ox, 0, width - 1);
            const int s_idx = (sy * width + sx) * 4;
            const int p_idx = ((oy + R) * side + (ox + R)) * 4;
            const float a = (float)dst[s_idx + 3] * inv255;
            pickup[p_idx + 0] = (float)dst[s_idx + 0] * inv255 * a;
            pickup[p_idx + 1] = (float)dst[s_idx + 1] * inv255 * a;
            pickup[p_idx + 2] = (float)dst[s_idx + 2] * inv255 * a;
            pickup[p_idx + 3] = a;
        }
    }

    int min_x = width, min_y = height, max_x = -1, max_y = -1;
    const float dx = x1 - x0;
    const float dy = y1 - y0;
    const float dist = hypotf(dx, dy);
    const float step = fmaxf(1.0f, radius * 0.25f);
    const int n_steps = (dist < 1e-4f) ? 0 : (int)ceilf(dist / step);
    const float carry = 0.20f + 0.75f * str;

    for (int i = 1; i <= n_steps; ++i) {
        const float t = (float)i / (float)n_steps;
        const int cx = (int)roundf(x0 + dx * t);
        const int cy = (int)roundf(y0 + dy * t);

        for (int oy = -R; oy <= R; ++oy) {
            const int py = cy + oy;
            if (py < 0 || py >= height) continue;
            for (int ox = -R; ox <= R; ++ox) {
                const int px = cx + ox;
                if (px < 0 || px >= width) continue;

                const float d = sqrtf((float)(ox * ox + oy * oy));
                const float falloff = calc_radial_falloff(d, radius, hardness);
                if (falloff <= 0.0f) continue;

                float sel = 1.0f;
                if (selection_mask) {
                    sel = (float)selection_mask[py * width + px] * inv255;
                    if (sel <= 0.0f) continue;
                }

                const float blend = clamp_float(str * falloff * sel, 0.0f, 1.0f);
                const int d_idx = (py * width + px) * 4;
                const int p_idx = ((oy + R) * side + (ox + R)) * 4;

                const float da = (float)dst[d_idx + 3] * inv255;
                const float dr_pm = (float)dst[d_idx + 0] * inv255 * da;
                const float dg_pm = (float)dst[d_idx + 1] * inv255 * da;
                const float db_pm = (float)dst[d_idx + 2] * inv255 * da;

                const float pr_pm = pickup[p_idx + 0];
                const float pg_pm = pickup[p_idx + 1];
                const float pb_pm = pickup[p_idx + 2];
                const float pa = pickup[p_idx + 3];

                const float nr_pm = dr_pm * (1.0f - blend) + pr_pm * blend;
                const float ng_pm = dg_pm * (1.0f - blend) + pg_pm * blend;
                const float nb_pm = db_pm * (1.0f - blend) + pb_pm * blend;
                const float na = da * (1.0f - blend) + pa * blend;

                if (na <= 1e-6f) {
                    dst[d_idx + 0] = 0;
                    dst[d_idx + 1] = 0;
                    dst[d_idx + 2] = 0;
                    dst[d_idx + 3] = 0;
                } else {
                    const float inv_na = 1.0f / na;
                    dst[d_idx + 0] = float_to_u8(nr_pm * inv_na * 255.0f);
                    dst[d_idx + 1] = float_to_u8(ng_pm * inv_na * 255.0f);
                    dst[d_idx + 2] = float_to_u8(nb_pm * inv_na * 255.0f);
                    dst[d_idx + 3] = float_to_u8(na * 255.0f);
                }

                pickup[p_idx + 0] = pr_pm * carry + dr_pm * (1.0f - carry);
                pickup[p_idx + 1] = pg_pm * carry + dg_pm * (1.0f - carry);
                pickup[p_idx + 2] = pb_pm * carry + db_pm * (1.0f - carry);
                pickup[p_idx + 3] = pa * carry + da * (1.0f - carry);

                if (px < min_x) min_x = px;
                if (py < min_y) min_y = py;
                if (px > max_x) max_x = px;
                if (py > max_y) max_y = py;
            }
        }
    }

    free(pickup);
    write_dirty_rect(out_dirty_rect, min_x, min_y, max_x, max_y);
}

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
) {
    if (!dst || width <= 0 || height <= 0 || radius <= 0.0f) {
        write_dirty_rect(out_dirty_rect, 0, 0, -1, -1);
        return;
    }
    const float str = clamp_float(strength, 0.0f, 1.0f);
    if (str <= 0.0f) {
        write_dirty_rect(out_dirty_rect, 0, 0, -1, -1);
        return;
    }

    const int k_rad = (kernel_size >= 5) ? 2 : 1;
    const int x0 = clamp_int((int)floorf(cx - radius), 0, width - 1);
    const int y0 = clamp_int((int)floorf(cy - radius), 0, height - 1);
    const int x1 = clamp_int((int)ceilf(cx + radius), 0, width - 1);
    const int y1 = clamp_int((int)ceilf(cy + radius), 0, height - 1);
    if (x1 < x0 || y1 < y0) {
        write_dirty_rect(out_dirty_rect, 0, 0, -1, -1);
        return;
    }

    const int patch_w = x1 - x0 + 1;
    const int patch_h = y1 - y0 + 1;
    float* blurred = (float*)malloc((size_t)patch_w * (size_t)patch_h * 4 * sizeof(float));
    if (!blurred) {
        write_dirty_rect(out_dirty_rect, 0, 0, -1, -1);
        return;
    }

    const float weights_3[3] = {1.0f, 2.0f, 1.0f};
    const float weights_5[5] = {1.0f, 4.0f, 6.0f, 4.0f, 1.0f};
    const float* weights = (k_rad == 2) ? weights_5 : weights_3;
    const float inv255 = 1.0f / 255.0f;

    /* 2D convolution on premultiplied RGBA */
    for (int py = y0; py <= y1; ++py) {
        for (int px = x0; px <= x1; ++px) {
            float sum_r = 0.0f, sum_g = 0.0f, sum_b = 0.0f, sum_a = 0.0f, sum_w = 0.0f;
            for (int ky = -k_rad; ky <= k_rad; ++ky) {
                const int sy = clamp_int(py + ky, 0, height - 1);
                const float wy = weights[ky + k_rad];
                for (int kx = -k_rad; kx <= k_rad; ++kx) {
                    const int sx = clamp_int(px + kx, 0, width - 1);
                    const float w = wy * weights[kx + k_rad];
                    const int s_idx = (sy * width + sx) * 4;
                    const float a = (float)dst[s_idx + 3] * inv255;
                    sum_r += (float)dst[s_idx + 0] * inv255 * a * w;
                    sum_g += (float)dst[s_idx + 1] * inv255 * a * w;
                    sum_b += (float)dst[s_idx + 2] * inv255 * a * w;
                    sum_a += a * w;
                    sum_w += w;
                }
            }
            const int b_idx = ((py - y0) * patch_w + (px - x0)) * 4;
            const float inv_w = 1.0f / sum_w;
            blurred[b_idx + 0] = sum_r * inv_w;
            blurred[b_idx + 1] = sum_g * inv_w;
            blurred[b_idx + 2] = sum_b * inv_w;
            blurred[b_idx + 3] = sum_a * inv_w;
        }
    }

    int min_x = width, min_y = height, max_x = -1, max_y = -1;
    for (int py = y0; py <= y1; ++py) {
        const float dy = (float)py - cy;
        for (int px = x0; px <= x1; ++px) {
            const float dx = (float)px - cx;
            const float dist = sqrtf(dx * dx + dy * dy);
            const float falloff = calc_radial_falloff(dist, radius, hardness);
            if (falloff <= 0.0f) continue;

            float sel = 1.0f;
            if (selection_mask) {
                sel = (float)selection_mask[py * width + px] * inv255;
                if (sel <= 0.0f) continue;
            }

            const float blend = clamp_float(str * falloff * sel, 0.0f, 1.0f);
            const int d_idx = (py * width + px) * 4;
            const int b_idx = ((py - y0) * patch_w + (px - x0)) * 4;

            const float da = (float)dst[d_idx + 3] * inv255;
            const float dr_pm = (float)dst[d_idx + 0] * inv255 * da;
            const float dg_pm = (float)dst[d_idx + 1] * inv255 * da;
            const float db_pm = (float)dst[d_idx + 2] * inv255 * da;

            const float br_pm = blurred[b_idx + 0];
            const float bg_pm = blurred[b_idx + 1];
            const float bb_pm = blurred[b_idx + 2];
            const float ba = blurred[b_idx + 3];

            float out_r_pm, out_g_pm, out_b_pm, out_a;
            if (mode == FILTER_MODE_SHARPEN) {
                const float amount = blend * 1.5f;
                out_r_pm = clamp_float(dr_pm + amount * (dr_pm - br_pm), 0.0f, 1.0f);
                out_g_pm = clamp_float(dg_pm + amount * (dg_pm - bg_pm), 0.0f, 1.0f);
                out_b_pm = clamp_float(db_pm + amount * (db_pm - bb_pm), 0.0f, 1.0f);
                out_a = da;
            } else {
                out_r_pm = dr_pm * (1.0f - blend) + br_pm * blend;
                out_g_pm = dg_pm * (1.0f - blend) + bg_pm * blend;
                out_b_pm = db_pm * (1.0f - blend) + bb_pm * blend;
                out_a = da * (1.0f - blend) + ba * blend;
            }

            if (out_a <= 1e-6f) {
                dst[d_idx + 0] = 0;
                dst[d_idx + 1] = 0;
                dst[d_idx + 2] = 0;
                dst[d_idx + 3] = 0;
            } else {
                const float inv_oa = 1.0f / out_a;
                dst[d_idx + 0] = float_to_u8(clamp_float(out_r_pm * inv_oa, 0.0f, 1.0f) * 255.0f);
                dst[d_idx + 1] = float_to_u8(clamp_float(out_g_pm * inv_oa, 0.0f, 1.0f) * 255.0f);
                dst[d_idx + 2] = float_to_u8(clamp_float(out_b_pm * inv_oa, 0.0f, 1.0f) * 255.0f);
                dst[d_idx + 3] = float_to_u8(clamp_float(out_a, 0.0f, 1.0f) * 255.0f);
            }

            if (px < min_x) min_x = px;
            if (py < min_y) min_y = py;
            if (px > max_x) max_x = px;
            if (py > max_y) max_y = py;
        }
    }

    free(blurred);
    write_dirty_rect(out_dirty_rect, min_x, min_y, max_x, max_y);
}

EXPORT void gaussian_blur_rgba(
    const uint8_t* src,
    uint8_t* dst,
    int width,
    int height,
    int radius
) {
    if (!src || !dst || width <= 0 || height <= 0) return;
    if (radius <= 0) {
        if (src != dst) {
            memcpy(dst, src, (size_t)width * (size_t)height * 4);
        }
        return;
    }

    const int r = clamp_int(radius, 1, 64);
    const int k_size = 2 * r + 1;
    float* kernel = (float*)malloc((size_t)k_size * sizeof(float));
    float* temp = (float*)malloc((size_t)width * (size_t)height * 4 * sizeof(float));
    if (!kernel || !temp) {
        free(kernel);
        free(temp);
        return;
    }

    const float sigma = fmaxf(0.5f, (float)r * 0.5f);
    const float inv_2sigma2 = 1.0f / (2.0f * sigma * sigma);
    float k_sum = 0.0f;
    for (int i = -r; i <= r; ++i) {
        const float w = expf(-(float)(i * i) * inv_2sigma2);
        kernel[i + r] = w;
        k_sum += w;
    }
    for (int i = 0; i < k_size; ++i) {
        kernel[i] /= k_sum;
    }

    const float inv255 = 1.0f / 255.0f;
    int y;

    /* Horizontal pass into premultiplied float temp */
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) if((int64_t)width * height > 16384)
    #endif
    for (y = 0; y < height; ++y) {
        const int row_off = y * width;
        for (int x = 0; x < width; ++x) {
            float sr = 0.0f, sg = 0.0f, sb = 0.0f, sa = 0.0f;
            for (int k = -r; k <= r; ++k) {
                const int sx = clamp_int(x + k, 0, width - 1);
                const float w = kernel[k + r];
                const int s_idx = (row_off + sx) * 4;
                const float a = (float)src[s_idx + 3] * inv255;
                sr += (float)src[s_idx + 0] * inv255 * a * w;
                sg += (float)src[s_idx + 1] * inv255 * a * w;
                sb += (float)src[s_idx + 2] * inv255 * a * w;
                sa += a * w;
            }
            const int t_idx = (row_off + x) * 4;
            temp[t_idx + 0] = sr;
            temp[t_idx + 1] = sg;
            temp[t_idx + 2] = sb;
            temp[t_idx + 3] = sa;
        }
    }

    /* Vertical pass back to straight uint8 dst */
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) if((int64_t)width * height > 16384)
    #endif
    for (y = 0; y < height; ++y) {
        const int row_off = y * width;
        for (int x = 0; x < width; ++x) {
            float sr = 0.0f, sg = 0.0f, sb = 0.0f, sa = 0.0f;
            for (int k = -r; k <= r; ++k) {
                const int sy = clamp_int(y + k, 0, height - 1);
                const float w = kernel[k + r];
                const int t_idx = (sy * width + x) * 4;
                sr += temp[t_idx + 0] * w;
                sg += temp[t_idx + 1] * w;
                sb += temp[t_idx + 2] * w;
                sa += temp[t_idx + 3] * w;
            }
            const int d_idx = (row_off + x) * 4;
            if (sa <= 1e-6f) {
                dst[d_idx + 0] = 0;
                dst[d_idx + 1] = 0;
                dst[d_idx + 2] = 0;
                dst[d_idx + 3] = 0;
            } else {
                const float inv_sa = 1.0f / sa;
                dst[d_idx + 0] = float_to_u8(clamp_float(sr * inv_sa, 0.0f, 1.0f) * 255.0f);
                dst[d_idx + 1] = float_to_u8(clamp_float(sg * inv_sa, 0.0f, 1.0f) * 255.0f);
                dst[d_idx + 2] = float_to_u8(clamp_float(sb * inv_sa, 0.0f, 1.0f) * 255.0f);
                dst[d_idx + 3] = float_to_u8(clamp_float(sa, 0.0f, 1.0f) * 255.0f);
            }
        }
    }

    free(kernel);
    free(temp);
}

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
) {
    if (!rgba || width <= 0 || height <= 0) return;

    const float tol = clamp_float(tolerance, 0.0f, 255.0f);
    const float soft = clamp_float(softness, 0.0f, 255.0f);
    const float desp = clamp_float(despill, 0.0f, 1.0f);
    const float inv_sqrt3 = 0.577350269f;
    const int64_t total = (int64_t)width * (int64_t)height;

    int dominant = 1; /* 0 = R, 1 = G, 2 = B */
    if (key_r >= key_g && key_r >= key_b) {
        dominant = 0;
    } else if (key_b >= key_r && key_b >= key_g) {
        dominant = 2;
    }

    int64_t i;
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(total > 32768)
    #endif
    for (i = 0; i < total; ++i) {
        const int64_t idx = i * 4;
        const uint8_t orig_a = rgba[idx + 3];
        if (orig_a == 0) continue;

        float r = (float)rgba[idx + 0];
        float g = (float)rgba[idx + 1];
        float b = (float)rgba[idx + 2];

        const float dr = r - (float)key_r;
        const float dg = g - (float)key_g;
        const float db = b - (float)key_b;
        const float dist = sqrtf(dr * dr + dg * dg + db * db) * inv_sqrt3;

        float matte = 1.0f;
        if (dist <= tol) {
            matte = 0.0f;
        } else if (soft > 1e-4f && dist < tol + soft) {
            const float t = (dist - tol) / soft;
            matte = t * t * (3.0f - 2.0f * t);
        }

        const uint8_t new_a = float_to_u8((float)orig_a * matte);
        rgba[idx + 3] = new_a;

        if (desp > 0.0f && new_a > 0) {
            if (dominant == 1) {
                const float limit = fmaxf(r, b);
                if (g > limit) {
                    g -= desp * (g - limit);
                    rgba[idx + 1] = float_to_u8(g);
                }
            } else if (dominant == 2) {
                const float limit = fmaxf(r, g);
                if (b > limit) {
                    b -= desp * (b - limit);
                    rgba[idx + 2] = float_to_u8(b);
                }
            } else {
                const float limit = fmaxf(g, b);
                if (r > limit) {
                    r -= desp * (r - limit);
                    rgba[idx + 0] = float_to_u8(r);
                }
            }
        }
    }
}

static inline int color_within_tolerance(
    const uint8_t* px,
    uint8_t sr,
    uint8_t sg,
    uint8_t sb,
    uint8_t sa,
    int tol
) {
    const int dr = abs((int)px[0] - (int)sr);
    const int dg = abs((int)px[1] - (int)sg);
    const int db = abs((int)px[2] - (int)sb);
    const int da = abs((int)px[3] - (int)sa);
    int m = dr > dg ? dr : dg;
    if (db > m) m = db;
    if (da > m) m = da;
    return m <= tol;
}

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
) {
    if (!rgba || width <= 0 || height <= 0 ||
        seed_x < 0 || seed_x >= width || seed_y < 0 || seed_y >= height) {
        write_dirty_rect(out_dirty_rect, 0, 0, -1, -1);
        return 0;
    }

    const int seed_pos = seed_y * width + seed_x;
    if (selection_mask && selection_mask[seed_pos] == 0) {
        write_dirty_rect(out_dirty_rect, 0, 0, -1, -1);
        return 0;
    }

    const int tol = clamp_int(tolerance, 0, 255);
    const uint8_t sr = rgba[seed_pos * 4 + 0];
    const uint8_t sg = rgba[seed_pos * 4 + 1];
    const uint8_t sb = rgba[seed_pos * 4 + 2];
    const uint8_t sa = rgba[seed_pos * 4 + 3];

    int min_x = width, min_y = height, max_x = -1, max_y = -1;
    int filled_count = 0;

    if (!contiguous) {
        for (int y = 0; y < height; ++y) {
            const int row_off = y * width;
            for (int x = 0; x < width; ++x) {
                const int pos = row_off + x;
                if (selection_mask && selection_mask[pos] == 0) continue;
                uint8_t* px = &rgba[pos * 4];
                if (color_within_tolerance(px, sr, sg, sb, sa, tol)) {
                    px[0] = fill_r;
                    px[1] = fill_g;
                    px[2] = fill_b;
                    px[3] = fill_a;
                    if (x < min_x) min_x = x;
                    if (y < min_y) min_y = y;
                    if (x > max_x) max_x = x;
                    if (y > max_y) max_y = y;
                    ++filled_count;
                }
            }
        }
        write_dirty_rect(out_dirty_rect, min_x, min_y, max_x, max_y);
        return filled_count;
    }

    const size_t total = (size_t)width * (size_t)height;
    uint8_t* visited = (uint8_t*)calloc(total, 1);
    int32_t* queue = (int32_t*)malloc(total * sizeof(int32_t));
    if (!visited || !queue) {
        free(visited);
        free(queue);
        write_dirty_rect(out_dirty_rect, 0, 0, -1, -1);
        return 0;
    }

    size_t head = 0;
    size_t tail = 0;
    visited[seed_pos] = 1;
    queue[tail++] = seed_pos;

    while (head < tail) {
        const int pos = queue[head++];
        const int x = pos % width;
        const int y = pos / width;
        uint8_t* px = &rgba[pos * 4];
        px[0] = fill_r;
        px[1] = fill_g;
        px[2] = fill_b;
        px[3] = fill_a;

        if (x < min_x) min_x = x;
        if (y < min_y) min_y = y;
        if (x > max_x) max_x = x;
        if (y > max_y) max_y = y;
        ++filled_count;

        const int nx_list[4] = {x - 1, x + 1, x, x};
        const int ny_list[4] = {y, y, y - 1, y + 1};
        for (int d = 0; d < 4; ++d) {
            const int nx = nx_list[d];
            const int ny = ny_list[d];
            if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
            const int npos = ny * width + nx;
            if (visited[npos]) continue;
            if (selection_mask && selection_mask[npos] == 0) continue;
            if (color_within_tolerance(&rgba[npos * 4], sr, sg, sb, sa, tol)) {
                visited[npos] = 1;
                queue[tail++] = npos;
            }
        }
    }

    free(visited);
    free(queue);
    write_dirty_rect(out_dirty_rect, min_x, min_y, max_x, max_y);
    return filled_count;
}

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
) {
    if (!rgba || !out_mask || width <= 0 || height <= 0 ||
        seed_x < 0 || seed_x >= width || seed_y < 0 || seed_y >= height) {
        write_dirty_rect(out_dirty_rect, 0, 0, -1, -1);
        return 0;
    }

    const size_t total = (size_t)width * (size_t)height;
    memset(out_mask, 0, total);

    const int tol = clamp_int(tolerance, 0, 255);
    const int seed_pos = seed_y * width + seed_x;
    const uint8_t sr = rgba[seed_pos * 4 + 0];
    const uint8_t sg = rgba[seed_pos * 4 + 1];
    const uint8_t sb = rgba[seed_pos * 4 + 2];
    const uint8_t sa = rgba[seed_pos * 4 + 3];

    int min_x = width, min_y = height, max_x = -1, max_y = -1;
    int selected_count = 0;

    if (!contiguous) {
        for (int y = 0; y < height; ++y) {
            const int row_off = y * width;
            for (int x = 0; x < width; ++x) {
                const int pos = row_off + x;
                if (color_within_tolerance(&rgba[pos * 4], sr, sg, sb, sa, tol)) {
                    out_mask[pos] = 255;
                    if (x < min_x) min_x = x;
                    if (y < min_y) min_y = y;
                    if (x > max_x) max_x = x;
                    if (y > max_y) max_y = y;
                    ++selected_count;
                }
            }
        }
        write_dirty_rect(out_dirty_rect, min_x, min_y, max_x, max_y);
        return selected_count;
    }

    int32_t* queue = (int32_t*)malloc(total * sizeof(int32_t));
    if (!queue) {
        write_dirty_rect(out_dirty_rect, 0, 0, -1, -1);
        return 0;
    }

    size_t head = 0;
    size_t tail = 0;
    out_mask[seed_pos] = 255;
    queue[tail++] = seed_pos;

    while (head < tail) {
        const int pos = queue[head++];
        const int x = pos % width;
        const int y = pos / width;

        if (x < min_x) min_x = x;
        if (y < min_y) min_y = y;
        if (x > max_x) max_x = x;
        if (y > max_y) max_y = y;
        ++selected_count;

        const int nx_list[4] = {x - 1, x + 1, x, x};
        const int ny_list[4] = {y, y, y - 1, y + 1};
        for (int d = 0; d < 4; ++d) {
            const int nx = nx_list[d];
            const int ny = ny_list[d];
            if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
            const int npos = ny * width + nx;
            if (out_mask[npos]) continue;
            if (color_within_tolerance(&rgba[npos * 4], sr, sg, sb, sa, tol)) {
                out_mask[npos] = 255;
                queue[tail++] = npos;
            }
        }
    }

    free(queue);
    write_dirty_rect(out_dirty_rect, min_x, min_y, max_x, max_y);
    return selected_count;
}

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
) {
    if (!rgba || width <= 0 || height <= 0) return 0;

    const size_t total = (size_t)width * (size_t)height;
    int32_t* labels = out_labels ? out_labels : (int32_t*)calloc(total, sizeof(int32_t));
    int32_t* queue = (int32_t*)malloc(total * sizeof(int32_t));
    if (!labels || !queue) {
        if (!out_labels) free(labels);
        free(queue);
        return 0;
    }
    if (out_labels) {
        memset(out_labels, 0, total * sizeof(int32_t));
    }

    size_t map_cap = 1024;
    int32_t* raw_to_final = (int32_t*)calloc(map_cap, sizeof(int32_t));
    if (!raw_to_final) {
        if (!out_labels) free(labels);
        free(queue);
        return 0;
    }

    int raw_id = 0;
    int stored_count = 0;
    const int min_a = min_area < 1 ? 1 : min_area;
    const int pad = padding < 0 ? 0 : padding;

    static const int dx8[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    static const int dy8[8] = {-1, -1, -1, 0, 0, 1, 1, 1};

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const int start_pos = y * width + x;
            if (labels[start_pos] != 0) continue;
            if (rgba[start_pos * 4 + 3] <= alpha_threshold) continue;

            ++raw_id;
            if ((size_t)raw_id >= map_cap) {
                size_t new_cap = map_cap * 2;
                int32_t* resized = (int32_t*)realloc(raw_to_final, new_cap * sizeof(int32_t));
                if (!resized) {
                    free(raw_to_final);
                    if (!out_labels) free(labels);
                    free(queue);
                    return stored_count;
                }
                memset(resized + map_cap, 0, (new_cap - map_cap) * sizeof(int32_t));
                raw_to_final = resized;
                map_cap = new_cap;
            }

            size_t head = 0;
            size_t tail = 0;
            labels[start_pos] = raw_id;
            queue[tail++] = start_pos;

            int min_x = x, min_y = y, max_x = x, max_y = y;
            int area = 0;

            while (head < tail) {
                const int pos = queue[head++];
                const int px = pos % width;
                const int py = pos / width;
                ++area;

                if (px < min_x) min_x = px;
                if (py < min_y) min_y = py;
                if (px > max_x) max_x = px;
                if (py > max_y) max_y = py;

                for (int d = 0; d < 8; ++d) {
                    const int nx = px + dx8[d];
                    const int ny = py + dy8[d];
                    if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
                    const int npos = ny * width + nx;
                    if (labels[npos] != 0) continue;
                    if (rgba[npos * 4 + 3] > alpha_threshold) {
                        labels[npos] = raw_id;
                        queue[tail++] = npos;
                    }
                }
            }

            if (area >= min_a) {
                if (out_boxes == NULL || stored_count < max_boxes) {
                    const int final_label = stored_count + 1;
                    raw_to_final[raw_id] = final_label;
                    if (out_boxes != NULL) {
                        const int bx0 = clamp_int(min_x - pad, 0, width - 1);
                        const int by0 = clamp_int(min_y - pad, 0, height - 1);
                        const int bx1 = clamp_int(max_x + pad, 0, width - 1);
                        const int by1 = clamp_int(max_y + pad, 0, height - 1);
                        out_boxes[stored_count].x = bx0;
                        out_boxes[stored_count].y = by0;
                        out_boxes[stored_count].width = bx1 - bx0 + 1;
                        out_boxes[stored_count].height = by1 - by0 + 1;
                        out_boxes[stored_count].area = area;
                        out_boxes[stored_count].label = final_label;
                    }
                    ++stored_count;
                }
            }
        }
    }

    if (out_labels) {
        for (size_t i = 0; i < total; ++i) {
            const int32_t raw = out_labels[i];
            if (raw > 0) {
                out_labels[i] = raw_to_final[raw];
            }
        }
    }

    free(raw_to_final);
    if (!out_labels) free(labels);
    free(queue);
    return stored_count;
}

EXPORT int find_alpha_bounds(
    const uint8_t* rgba,
    int width,
    int height,
    uint8_t alpha_threshold,
    int* out_rect
) {
    if (!rgba || width <= 0 || height <= 0) {
        write_dirty_rect(out_rect, 0, 0, -1, -1);
        return 0;
    }

    int min_x = width, min_y = height, max_x = -1, max_y = -1;
    for (int y = 0; y < height; ++y) {
        const int row_off = y * width;
        for (int x = 0; x < width; ++x) {
            if (rgba[(row_off + x) * 4 + 3] > alpha_threshold) {
                if (x < min_x) min_x = x;
                if (y < min_y) min_y = y;
                if (x > max_x) max_x = x;
                if (y > max_y) max_y = y;
            }
        }
    }

    if (max_x < min_x || max_y < min_y) {
        write_dirty_rect(out_rect, 0, 0, -1, -1);
        return 0;
    }

    write_dirty_rect(out_rect, min_x, min_y, max_x, max_y);
    return 1;
}

EXPORT void bilinear_resize_rgba(
    const uint8_t* src,
    int src_w,
    int src_h,
    uint8_t* dst,
    int dst_w,
    int dst_h
) {
    if (!src || !dst || src_w <= 0 || src_h <= 0 || dst_w <= 0 || dst_h <= 0) return;

    const float scale_x = (float)src_w / (float)dst_w;
    const float scale_y = (float)src_h / (float)dst_h;
    const float inv255 = 1.0f / 255.0f;

    int dy;
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) if((int64_t)dst_w * dst_h > 16384)
    #endif
    for (dy = 0; dy < dst_h; ++dy) {
        const float fy = clamp_float(((float)dy + 0.5f) * scale_y - 0.5f, 0.0f, (float)(src_h - 1));
        const int y0 = (int)floorf(fy);
        const int y1 = clamp_int(y0 + 1, 0, src_h - 1);
        const float wy = fy - (float)y0;
        const float wy0 = 1.0f - wy;

        for (int dx = 0; dx < dst_w; ++dx) {
            const float fx = clamp_float(((float)dx + 0.5f) * scale_x - 0.5f, 0.0f, (float)(src_w - 1));
            const int x0 = (int)floorf(fx);
            const int x1 = clamp_int(x0 + 1, 0, src_w - 1);
            const float wx = fx - (float)x0;
            const float wx0 = 1.0f - wx;

            const float w00 = wx0 * wy0;
            const float w10 = wx * wy0;
            const float w01 = wx0 * wy;
            const float w11 = wx * wy;

            const int i00 = (y0 * src_w + x0) * 4;
            const int i10 = (y0 * src_w + x1) * 4;
            const int i01 = (y1 * src_w + x0) * 4;
            const int i11 = (y1 * src_w + x1) * 4;

            const float a00 = (float)src[i00 + 3] * inv255;
            const float a10 = (float)src[i10 + 3] * inv255;
            const float a01 = (float)src[i01 + 3] * inv255;
            const float a11 = (float)src[i11 + 3] * inv255;

            const float out_a = a00 * w00 + a10 * w10 + a01 * w01 + a11 * w11;
            const int d_idx = (dy * dst_w + dx) * 4;

            if (out_a <= 1e-6f) {
                dst[d_idx + 0] = 0;
                dst[d_idx + 1] = 0;
                dst[d_idx + 2] = 0;
                dst[d_idx + 3] = 0;
                continue;
            }

            const float r_pm = ((float)src[i00 + 0] * a00 * w00 +
                                (float)src[i10 + 0] * a10 * w10 +
                                (float)src[i01 + 0] * a01 * w01 +
                                (float)src[i11 + 0] * a11 * w11) * inv255;
            const float g_pm = ((float)src[i00 + 1] * a00 * w00 +
                                (float)src[i10 + 1] * a10 * w10 +
                                (float)src[i01 + 1] * a01 * w01 +
                                (float)src[i11 + 1] * a11 * w11) * inv255;
            const float b_pm = ((float)src[i00 + 2] * a00 * w00 +
                                (float)src[i10 + 2] * a10 * w10 +
                                (float)src[i01 + 2] * a01 * w01 +
                                (float)src[i11 + 2] * a11 * w11) * inv255;

            const float inv_oa = 1.0f / out_a;
            dst[d_idx + 0] = float_to_u8(clamp_float(r_pm * inv_oa, 0.0f, 1.0f) * 255.0f);
            dst[d_idx + 1] = float_to_u8(clamp_float(g_pm * inv_oa, 0.0f, 1.0f) * 255.0f);
            dst[d_idx + 2] = float_to_u8(clamp_float(b_pm * inv_oa, 0.0f, 1.0f) * 255.0f);
            dst[d_idx + 3] = float_to_u8(clamp_float(out_a, 0.0f, 1.0f) * 255.0f);
        }
    }
}
