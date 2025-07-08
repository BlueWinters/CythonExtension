
#include <omp.h>
#include <immintrin.h>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>
#include "format.h"


struct InterpolationIndex {
    int y0, x0, y1, x1;
    float ly, lx, hy, hx;
};

struct InterpolationIndexBuffer {
    int src_h, src_w, dst_h, dst_w;
    std::vector<InterpolationIndex> index;
};

std::vector<InterpolationIndexBuffer> IndexBufferVector;


InterpolationIndexBuffer& buildInterpolateIndex(
    int src_h,
    int src_w,
    int dst_h,
    int dst_w,
    std::vector<InterpolationIndexBuffer>& index_buffer_vector
)
{
    for (int n = 0; n < index_buffer_vector.size(); n++) {
        InterpolationIndexBuffer& buffer = index_buffer_vector[n];
        if (buffer.src_h == src_h && buffer.src_w == src_w &&
            buffer.dst_h == dst_h && buffer.dst_w == dst_w) {
            return buffer;
        }
    }

    index_buffer_vector.push_back(InterpolationIndexBuffer());
    InterpolationIndexBuffer& buffer = index_buffer_vector.back();
    buffer.src_h = src_h;
    buffer.src_w = src_w;
    buffer.dst_h = dst_h;
    buffer.dst_w = dst_w;
    std::vector<InterpolationIndex>& index = buffer.index;
    index.resize(dst_h * dst_w * 3);

    float scale_y = static_cast<float>(src_h) / dst_h;
    float scale_x = static_cast<float>(src_w) / dst_w;
    for (int y = 0; y < dst_h; ++y) {
        for (int x = 0; x < dst_w; ++x) {
            float src_fy = (y + 0.5f) * scale_y - 0.5f;
            float src_fx = (x + 0.5f) * scale_x - 0.5f;
            int y0 = static_cast<int>(floorf(src_fy));
            int x0 = static_cast<int>(floorf(src_fx));
            int y1 = std::min(y0 + 1, src_h - 1);
            int x1 = std::min(x0 + 1, src_w - 1);
            float ly = src_fy - y0;
            float lx = src_fx - x0;
            float hy = 1.0f - ly;
            float hx = 1.0f - lx;
            y0 = std::max(y0, 0);
            x0 = std::max(x0, 0);
            index[y * dst_w + x] = {y0, x0, y1, x1, ly, lx, hy, hx};
        }
    }

    return buffer;
}

void resizeAndTranspose(
    const unsigned char* src,
    int src_h,
    int src_w,
    float* dst,
    int dst_h,
    int dst_w
)
{
    InterpolationIndexBuffer& index_buffer = buildInterpolateIndex(
        src_h, src_w, dst_h, dst_w, IndexBufferVector);
    std::vector<InterpolationIndex>& index_vector = index_buffer.index;

    #pragma omp parallel for collapse(3)
    for (int c = 0; c < 3; ++c) {
        for (int idx = 0; idx < dst_h * dst_w; ++idx) {
            int y = idx / dst_w;
            int x = idx % dst_w;
            const auto& index = index_vector[idx];
            int y0 = index.y0;
            int x0 = index.x0;
            int y1 = index.y1;
            int x1 = index.x1;
            float ly = index.ly;
            float lx = index.lx;
            float hy = index.hy;
            float hx = index.hx;

            float v00 = float(src[(y0 * src_w + x0) * 3 + c]);
            float v01 = float(src[(y0 * src_w + x1) * 3 + c]);
            float v10 = float(src[(y1 * src_w + x0) * 3 + c]);
            float v11 = float(src[(y1 * src_w + x1) * 3 + c]);

            float v0 = hx * v00 + lx * v01;
            float v1 = hx * v10 + lx * v11;
            float v = hy * v0 + ly * v1;

            dst[c * dst_h * dst_w + y * dst_w + x] = v;
        }
    }
}

void paddingAndNormalizes(
    float* fmt,
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
    const int rsz_size = rsz_h * rsz_w;
    float val0 = (float(padding_value) - mean0) * scale0;
    float val1 = (float(padding_value) - mean1) * scale1;
    float val2 = (float(padding_value) - mean2) * scale2;

    #pragma omp parallel for num_threads(3)
    for (int c = 0; c < 3; ++c) {
        float pad_val = (c == 0 ? val0 : (c == 1 ? val1 : val2));
        for (int i = 0; i < dst_size; ++i) {
            dst[c * dst_size + i] = pad_val;
        }
    }

    #pragma omp parallel for num_threads(3)
    for (int c = 0; c < 3; ++c) {
        float mean = (c == 0 ? mean0 : (c == 1 ? mean1 : mean2));
        float scale = (c == 0 ? scale0 : (c == 1 ? scale1 : scale2));
        for (int y = 0; y < rsz_h; ++y) {
            int dst_y = y + pad_top;
            for (int x = 0; x < rsz_w; ++x) {
                int dst_x = x + pad_lft;
                int dst_idx = c * dst_size + dst_y * dst_w + dst_x;
                int src_idx = c * rsz_h * rsz_w + y * rsz_w + x;
                dst[dst_idx] = (fmt[src_idx] - mean) * scale;
            }
        }
    }
}

int postprocess_indexing(
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
    float* fmt = new float[rsz_h * rsz_w * 3];
    resizeAndTranspose(src, src_h, src_w, fmt, rsz_h, rsz_w);
    paddingAndNormalizes(
        fmt, rsz_h, rsz_w, pad_lft, pad_top, pad_rig, pad_bot, padding_value,
        mean0, mean1, mean2, scale0, scale1, scale2, dst, dst_h, dst_w);
    delete[] fmt;
    return Format_Indexing;
}
