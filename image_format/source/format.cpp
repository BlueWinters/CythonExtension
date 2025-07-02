
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

int postprocess_openmp(
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

    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        dst[0 * size + i] = val0;
        dst[1 * size + i] = val1;
        dst[2 * size + i] = val2;
    }

    #pragma omp parallel for collapse(2)
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
    return Format_OpenMP;
}

int postprocess_openmp_avx2(
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
    if (isAVX2Supported() == false) {
        return Format_Error;
    }

    const int size = dst_h * dst_w;
    float val0 = (float(padding_value) - mean0) * scale0;
    float val1 = (float(padding_value) - mean1) * scale1;
    float val2 = (float(padding_value) - mean2) * scale2;

    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        dst[0 * size + i] = val0;
        dst[1 * size + i] = val1;
        dst[2 * size + i] = val2;
    }

    #pragma omp parallel for collapse(2)
    for (int y = 0; y < src_h; ++y) {
        for (int c = 0; c < 3; ++c) {
            float mean = (c == 0 ? mean0 : (c == 1 ? mean1 : mean2));
            float scale = (c == 0 ? scale0 : (c == 1 ? scale1 : scale2));
            __m256 vmean = _mm256_set1_ps(mean);
            __m256 vscale = _mm256_set1_ps(scale);

            int dst_y = y + pad_top;
            int x = 0;
            for (; x + 7 < src_w; x += 8) {
                float vals[8];
                for (int i = 0; i < 8; ++i) {
                    vals[i] = float(src[(y * src_w + x + i) * 3 + c]);
                }
                __m256 vsrc = _mm256_loadu_ps(vals);
                __m256 vres = _mm256_mul_ps(_mm256_sub_ps(vsrc, vmean), vscale);
                int dst_idx = c * dst_h * dst_w + dst_y * dst_w + (x + pad_lft);
                _mm256_storeu_ps(dst + dst_idx, vres);
            }
            // 处理剩余像素
            for (; x < src_w; ++x) {
                int src_idx = (y * src_w + x) * 3 + c;
                int dst_idx = c * dst_h * dst_w + dst_y * dst_w + (x + pad_lft);
                dst[dst_idx] = (float(src[src_idx]) - mean) * scale;
            }
        }
    }
    return Format_OpenMP_AVX2;
}

int postprocess(
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
    int dst_w,
    bool enable_avx2
)
{
    if (enable_avx2 == true) {
        return postprocess_openmp_avx2(
            src, src_h, src_w, 
            pad_lft, pad_top, 
            pad_rig, pad_bot, 
            mean0, mean1, mean2, 
            scale0, scale1, scale2, 
            padding_value,
            dst, dst_h, dst_w
        );
    } else {
        return postprocess_openmp(
            src, src_h, src_w, 
            pad_lft, pad_top, 
            pad_rig, pad_bot, 
            mean0, mean1, mean2, 
            scale0, scale1, scale2, 
            padding_value,
            dst, dst_h, dst_w
        );  
    }
}