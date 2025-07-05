
#include <omp.h>
#include <immintrin.h>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>
#include "format.h"


int postprocess_end2end(
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

     // 计算resize比例
     const float scale_y = float(src_h) / float(rsz_h);
     const float scale_x = float(src_w) / float(rsz_w);

     #pragma omp parallel for num_threads(3)
     for (int c = 0; c < 3; ++c) {
         const float val = (c == 0 ? val0 : (c == 1 ? val1 : val2));
         const float mean = (c == 0 ? mean0 : (c == 1 ? mean1 : mean2));
         const float scale = (c == 0 ? scale0 : (c == 1 ? scale1 : scale2));
         const int pre_size_c = c * dst_size;
         for (int y = 0; y < dst_h; ++y) {
             int pre_size_cy = pre_size_c + y * dst_w;
             for (int x = 0; x < dst_w; ++x) {
                 int dst_idx = pre_size_cy + x;  // c * dst_size + y * dst_w + x;
                 // 判断是否在有效区域
                 if (y >= pad_top && y < pad_top + rsz_h &&
                     x >= pad_lft && x < pad_lft + rsz_w) {
                     // 映射到resize后图像坐标
                     int rsz_y = y - pad_top;
                     int rsz_x = x - pad_lft;
                     // 映射到原图浮点坐标
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
                     // 边界保护
                     y0 = std::max(y0, 0);
                     x0 = std::max(x0, 0);
                     // 双线性插值
                     float v00 = float(src[(y0 * src_w + x0) * 3 + c]);
                     float v01 = float(src[(y0 * src_w + x1) * 3 + c]);
                     float v10 = float(src[(y1 * src_w + x0) * 3 + c]);
                     float v11 = float(src[(y1 * src_w + x1) * 3 + c]);
                     float v0 = hx * v00 + lx * v01;
                     float v1 = hx * v10 + lx * v11;
                     float v = roundf(hy * v0 + ly * v1);
                     // float v = hy * v0 + ly * v1;
                     // 归一化
                     dst[dst_idx] = (v - mean) * scale;
                 } else {
                     // padding 区域
                     dst[dst_idx] = val;
                 }
             }
         }
     }
     return Format_End2End;
}