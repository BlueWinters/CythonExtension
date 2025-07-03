
#include <omp.h>
#include <immintrin.h>
#include <cmath>
#include <iostream>
#include "format.h"


#if defined(_MSC_VER) // MSVC
#include <intrin.h>
bool isSSE2Supported() {
    int cpuInfo[4] = {0};
    __cpuid(cpuInfo, 1);
    return (cpuInfo[3] & (1 << 26)) != 0;
}
bool isAVX2Supported() {
    int cpuInfo[4] = {0};
    __cpuid(cpuInfo, 0);
    if (cpuInfo[0] < 7) return false;
    __cpuidex(cpuInfo, 7, 0);
    return (cpuInfo[1] & (1 << 5)) != 0;
}
#elif defined(__GNUC__) || defined(__clang__) // GCC/Clang
bool isSSE2Supported() {
#if defined(__x86_64__) || defined(_M_X64)
    return true; // 64位x86必有SSE2
#else
    return __builtin_cpu_supports("sse2");
#endif
}
bool isAVX2Supported() {
    return __builtin_cpu_supports("avx2");
}
#else
bool isSSE2Supported() { return false; }
bool isAVX2Supported() { return false; }
#endif


bool preprocess(
    int src_h, 
    int src_w, 
    int dst_h, 
    int dst_w, 
    int& rsz_h,
    int& rsz_w, 
    int& pad_lft,
    int& pad_rig,
    int& pad_top,
    int& pad_bot
)
{   
    if (src_w > 0 && src_h > 0 && dst_w > 0 && dst_h > 0) {
        float src_ratio = float(src_h) / float(src_w);
        float dst_ratio = float(dst_h) / float(dst_w);
        if (src_ratio > dst_ratio) {
            rsz_h = dst_h;
            rsz_w = int(std::round(float(src_w) / float(src_h) * dst_h));
            pad_lft = (dst_w - rsz_w) / 2;
            pad_rig = dst_w - rsz_w - pad_lft;
            pad_top = 0;
            pad_bot = 0;
        }
        else {
            rsz_w = dst_w;
            rsz_h = int(std::round(float(src_h) / float(src_w) * dst_w));
            pad_lft = 0;
            pad_rig = 0;
            pad_top = (dst_h - rsz_h) / 2;
            pad_bot = dst_h - rsz_h - pad_top;
        }
        return true;
    }
    else {
        return false;
    }
}

int postprocess_native(
    const unsigned char* src,
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

    for (int i = 0; i < size; ++i) {
        dst[0 * size + i] = val0;
        dst[1 * size + i] = val1;
        dst[2 * size + i] = val2;
    }

    for (int y = 0; y < src_h; ++y) {
        for (int x = 0; x < src_w; ++x) {
            int dst_y = y + pad_top;
            int dst_x = x + pad_lft;
            int src_idx = (y * src_w + x) * 3;
            int dst_idx0 = 0 * size + dst_y * dst_w + dst_x;
            int dst_idx1 = 1 * size + dst_y * dst_w + dst_x;
            int dst_idx2 = 2 * size + dst_y * dst_w + dst_x;
            dst[dst_idx0] = (float(src[src_idx + 0]) - mean0) * scale0;
            dst[dst_idx1] = (float(src[src_idx + 1]) - mean1) * scale1;
            dst[dst_idx2] = (float(src[src_idx + 2]) - mean2) * scale2;
        }
    }
    return Format_Native;
}

int postprocess_openmp1(
    const unsigned char* src,
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
    #pragma omp parallel for num_threads(3)
    for (int c = 0; c < 3; ++c) {
        float mean = (c == 0 ? mean0 : (c == 1 ? mean1 : mean2));
        float scale = (c == 0 ? scale0 : (c == 1 ? scale1 : scale2));
        float pad_val = (float(padding_value) - mean) * scale;
        for (int y = 0; y < dst_h; ++y) {
            for (int x = 0; x < dst_w; ++x) {
                int dst_idx = c * size + y * dst_w + x;
                // 判断是否在padding区
                if (y < pad_top || y >= pad_top + src_h || x < pad_lft || x >= pad_lft + src_w) {
                    dst[dst_idx] = pad_val;
                } else {
                    int src_y = y - pad_top;
                    int src_x = x - pad_lft;
                    int src_idx = (src_y * src_w + src_x) * 3 + c;
                    dst[dst_idx] = (float(src[src_idx]) - mean) * scale;
                }
            }
        }
    }
    return Format_OpenMP;
}

int postprocess_openmp2(
    const unsigned char* src,
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

    #pragma omp parallel for num_threads(3)
    for (int c = 0; c < 3; ++c) {
        int pre_size = c * size;
        float val = (c == 0 ? val0 : (c == 1 ? val1 : val2));
        for (int i = 0; i < size; ++i) {
            dst[pre_size + i] = val;
        }
    }

    #pragma omp parallel for num_threads(3)
    for (int c = 0; c < 3; ++c) {
        float mean = (c == 0 ? mean0 : (c == 1 ? mean1 : mean2));
        float scale = (c == 0 ? scale0 : (c == 1 ? scale1 : scale2));
        int pre_size_c = c * size;
        for (int y = 0; y < src_h; ++y) {
            int dst_y = y + pad_top;
            int pre_size_cy = pre_size_c + dst_y * dst_w;
            int pre_size_yw = y * src_w;
            for (int x = 0; x < src_w; ++x) {
                int dst_x = x + pad_lft;
                int src_idx = (pre_size_yw + x) * 3 + c;
                int dst_idx = pre_size_cy + dst_x;
                dst[dst_idx] = (float(src[src_idx]) - mean) * scale;
            }
        }
    }
    return Format_OpenMP;
}


int postprocess_openmp3(
    const unsigned char* src,
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

    #pragma omp parallel for num_threads(3)
    for (int c = 0; c < 3; ++c) {
        int pre_size = c * size;
        float val = (c == 0 ? val0 : (c == 1 ? val1 : val2));
        __m256 vval = _mm256_set1_ps(val);
        int i = 0;
        float* ptr = dst + pre_size;
        for (; i + 7 < size; i += 8) {
            _mm256_storeu_ps(ptr + i, vval);
        }
        for (; i < size; ++i) {
            ptr[i] = val;
        }
    }

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
    return Format_OpenMP;
}

int postprocess_full(
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
    return Format_OpenMP;
}
