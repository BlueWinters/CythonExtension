
#include <omp.h>
#include <immintrin.h>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>
#include "format.h"
#include "assign.h"


void resize(
    const unsigned char* src,
    int src_h,
    int src_w,
    float* dst,
    int dst_h,
    int dst_w
)
{
    const float scale_y = static_cast<float>(src_h) / dst_h;
    const float scale_x = static_cast<float>(src_w) / dst_w;

    #pragma omp parallel for collapse(2)
    for (int y = 0; y < dst_h; ++y) {
        for (int x = 0; x < dst_w; ++x) {
            float src_fy = (y + 0.5f) * scale_y - 0.5f;
            float src_fx = (x + 0.5f) * scale_x - 0.5f;
            int sy = static_cast<int>(std::floor(src_fy));
            int sx = static_cast<int>(std::floor(src_fx));
            float ly = src_fy - sy;
            float lx = src_fx - sx;
            sy = std::max(sy, 0);
            sx = std::max(sx, 0);
            int ey = std::min(sy + 1, src_h - 1);
            int ex = std::min(sx + 1, src_w - 1);
            for (int c = 0; c < 3; ++c) {
                float v00 = static_cast<float>(src[(sy * src_w + sx) * 3 + c]);
                float v01 = static_cast<float>(src[(sy * src_w + ex) * 3 + c]);
                float v10 = static_cast<float>(src[(ey * src_w + sx) * 3 + c]);
                float v11 = static_cast<float>(src[(ey * src_w + ex) * 3 + c]);
                float top = (1.0f - lx) * v00 + lx * v01;
                float bot = (1.0f - lx) * v10 + lx * v11;
                float val = (1.0f - ly) * top + ly * bot;
                dst[(y * dst_w + x) * 3 + c] = val;
            }
        }
    }
}

void normalize(
    const float* src,
    int src_h,
    int src_w,
    int pad_lft,
    int pad_top,
    int pad_rig, 
    int pad_bot,
    float mean0, 
    float mean1, 
    float mean2,
    float scale0,
    float scale1,
    float scale2,
    unsigned char padding_value,
    float* dst,
    int dst_h, 
    int dst_w
)
{
    const int size = dst_h * dst_w;
    float val0 = (float(padding_value) - mean0) * scale0;
    float val1 = (float(padding_value) - mean1) * scale1;
    float val2 = (float(padding_value) - mean2) * scale2;
    const float vals[3] = {val0, val1, val2};
    assignValue1(dst, dst_h, dst_w, vals);

    #pragma omp parallel for num_threads(3)
    for (int c = 0; c < 3; ++c) {
        float mean = (c == 0 ? mean0 : (c == 1 ? mean1 : mean2));
        float scale = (c == 0 ? scale0 : (c == 1 ? scale1 : scale2));
        const __m256 vmean = _mm256_set1_ps(mean);
        const __m256 vscale = _mm256_set1_ps(scale);
        int dst_idx_csize = c * size;
        for (int y = 0; y < src_h; ++y) {
            int dst_y = y + pad_top;
            int dst_dix_csize_yw = dst_idx_csize + dst_y * dst_w;
            int src_idx_yw = y * src_w;
            int x = 0;
            for (; x + 7 < src_w; x += 8) {
                int src_idx = (src_idx_yw + x) * 3 + c;
                int dst_idx = dst_dix_csize_yw + pad_lft + x;
                // 加载8个像素
                __m256 src_v = _mm256_set_ps(
                    float(src[src_idx + 7 * 3]),
                    float(src[src_idx + 6 * 3]),
                    float(src[src_idx + 5 * 3]),
                    float(src[src_idx + 4 * 3]),
                    float(src[src_idx + 3 * 3]),
                    float(src[src_idx + 2 * 3]),
                    float(src[src_idx + 1 * 3]),
                    float(src[src_idx + 0 * 3])
                );
                // 归一化
                src_v = _mm256_sub_ps(src_v, vmean);
                src_v = _mm256_mul_ps(src_v, vscale);
                _mm256_storeu_ps(dst + dst_idx, src_v);
            }
            // 剩余像素
            for (; x < src_w; ++x) {
                int src_idx = (src_idx_yw + x) * 3 + c;
                int dst_idx = dst_dix_csize_yw + pad_lft + x;
                dst[dst_idx] = (float(src[src_idx]) - mean) * scale;
            }
        }
    }
}

int postprocess_end2end_v1(
    const unsigned char* src,
    int src_h,
    int src_w,
    int rsz_h,
    int rsz_w,
    int pad_lft,
    int pad_top,
    int pad_rig,
    int pad_bot,
    unsigned char padding_value,
    float mean0,
    float mean1,
    float mean2,
    float scale0,
    float scale1,
    float scale2,
    float* dst,
    int dst_h,
    int dst_w
)
{
    float* resized = new float[rsz_h * rsz_w * 3]();
    resize(src, src_h, src_w, resized, rsz_h, rsz_w);
    normalize(resized, rsz_h, rsz_w, pad_lft, pad_top, pad_rig, pad_bot, 
        mean0, mean1, mean2, scale0, scale1, scale2, padding_value, dst, dst_h, dst_w);
    delete[] resized;
    return Format_End2End;
}


int postprocess_end2end_v2(
    const unsigned char* src,
    int src_h,
    int src_w,
    int rsz_h,
    int rsz_w,
    int pad_lft,
    int pad_top,
    int pad_rig,
    int pad_bot,
    unsigned char padding_value,
    float mean0,
    float mean1,
    float mean2,
    float scale0,
    float scale1,
    float scale2,
    float* dst,
    int dst_h,
    int dst_w
)
{
    const int dst_size = dst_h * dst_w;
    const float val0 = (float(padding_value) - mean0) * scale0;
    const float val1 = (float(padding_value) - mean1) * scale1;
    const float val2 = (float(padding_value) - mean2) * scale2;

    const float scale_y = float(src_h) / float(rsz_h);
    const float scale_x = float(src_w) / float(rsz_w);

    #pragma omp parallel for num_threads(3)
    for (int c = 0; c < 3; ++c) {
        const float val = (c == 0 ? val0 : (c == 1 ? val1 : val2));
        const float mean = (c == 0 ? mean0 : (c == 1 ? mean1 : mean2));
        const float scale = (c == 0 ? scale0 : (c == 1 ? scale1 : scale2));
        const int pre_size_c = c * dst_size;

        const int thread_count = std::min(4, omp_get_max_threads() / 3);
        #pragma omp parallel for num_threads(thread_count) collapse(2)
        for (int y = 0; y < dst_h; ++y) {
            for (int x = 0; x < dst_w; ++x) {
                int pre_size_cy = pre_size_c + y * dst_w;
                int dst_idx = pre_size_cy + x;  // c * dst_size + y * dst_w + x;
                if (y >= pad_top && y < pad_top + rsz_h &&
                    x >= pad_lft && x < pad_lft + rsz_w) {
                    int rsz_y = y - pad_top;
                    int rsz_x = x - pad_lft;
                    float src_fy = (rsz_y + 0.5f) * scale_y - 0.5f;
                    float src_fx = (rsz_x + 0.5f) * scale_x - 0.5f;
                    int y0 = int(floorf(src_fy));
                    int x0 = int(floorf(src_fx));
                    int y1 = std::min(y0 + 1, src_h - 1);
                    int x1 = std::min(x0 + 1, src_w - 1);
                    float ly = src_fy - y0;
                    float lx = src_fx - x0;
                    float hy = 1.0f - ly;
                    float hx = 1.0f - lx;
                    y0 = std::max(y0, 0);
                    x0 = std::max(x0, 0);
                    // interpolate
                    float v00 = float(src[(y0 * src_w + x0) * 3 + c]);
                    float v01 = float(src[(y0 * src_w + x1) * 3 + c]);
                    float v10 = float(src[(y1 * src_w + x0) * 3 + c]);
                    float v11 = float(src[(y1 * src_w + x1) * 3 + c]);
                    float v0 = hx * v00 + lx * v01;
                    float v1 = hx * v10 + lx * v11;
                    float v = roundf(hy * v0 + ly * v1);
                    // float v = hy * v0 + ly * v1;
                    dst[dst_idx] = (v - mean) * scale;
                }
               else {
                   // padding area
                   dst[dst_idx] = val;
               }
            }
        }
    }
    return Format_End2End;
}


