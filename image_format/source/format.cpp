
#include <omp.h>
#include <immintrin.h>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>
#include "format.h"
#include "assign.h"


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


int getVersionOpenMP() {
#ifdef _OPENMP
    return _OPENMP;
#else
    return 0;
#endif
}

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

int postprocess1_native(
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
    return Format1_Native;
}

int postprocess1_openmp(
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

    #pragma omp parallel for num_threads(4)
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
    return Format1_OpenMP;
}


int postprocess1_openmp_sse2(
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

    #pragma omp parallel for num_threads(4) schedule(static)
    for (int y = 0; y < src_h; ++y) {
        for (int x = 0; x + 3 < src_w; x += 4) {
            int dst_y = y + pad_top;
            int dst_x = x + pad_lft;
            int src_idx = (y * src_w + x) * 3;
            int dst_idx0 = 0 * size + dst_y * dst_w + dst_x;
            int dst_idx1 = 1 * size + dst_y * dst_w + dst_x;
            int dst_idx2 = 2 * size + dst_y * dst_w + dst_x;

            // 收集4个像素的每个通道
            float b_arr[4], g_arr[4], r_arr[4];
            for (int i = 0; i < 4; ++i) {
                b_arr[i] = float(src[src_idx + i * 3 + 0]);
                g_arr[i] = float(src[src_idx + i * 3 + 1]);
                r_arr[i] = float(src[src_idx + i * 3 + 2]);
            }
            __m128 b_f = _mm_loadu_ps(b_arr);
            __m128 g_f = _mm_loadu_ps(g_arr);
            __m128 r_f = _mm_loadu_ps(r_arr);

            b_f = _mm_mul_ps(_mm_sub_ps(b_f, _mm_set1_ps(mean0)), _mm_set1_ps(scale0));
            g_f = _mm_mul_ps(_mm_sub_ps(g_f, _mm_set1_ps(mean1)), _mm_set1_ps(scale1));
            r_f = _mm_mul_ps(_mm_sub_ps(r_f, _mm_set1_ps(mean2)), _mm_set1_ps(scale2));

            _mm_storeu_ps(dst + dst_idx0, b_f);
            _mm_storeu_ps(dst + dst_idx1, g_f);
            _mm_storeu_ps(dst + dst_idx2, r_f);
        }
        // 处理剩余像素
        for (int x = (src_w & ~3); x < src_w; ++x) {
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
    return Format_Error;
}

int postprocess1_openmp_avx2(
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
    // 填充padding区域
    float val0 = (float(padding_value) - mean0) * scale0;
    float val1 = (float(padding_value) - mean1) * scale1;
    float val2 = (float(padding_value) - mean2) * scale2;
    for (int i = 0; i < size; ++i) {
        dst[0 * size + i] = val0;
        dst[1 * size + i] = val1;
        dst[2 * size + i] = val2;
    }

    // 有效区域AVX2加速
    #pragma omp parallel for num_threads(4) schedule(static)
    for (int y = 0; y < src_h; ++y) {
        for (int x = 0; x + 7 < src_w; x += 8) {
            int dst_y = y + pad_top;
            int dst_x = x + pad_lft;
            int src_idx = (y * src_w + x) * 3;
            int dst_idx0 = 0 * size + dst_y * dst_w + dst_x;
            int dst_idx1 = 1 * size + dst_y * dst_w + dst_x;
            int dst_idx2 = 2 * size + dst_y * dst_w + dst_x;

            // 分别收集8个像素的每个通道
            float b_arr[8], g_arr[8], r_arr[8];
            for (int i = 0; i < 8; ++i) {
                b_arr[i] = float(src[src_idx + i * 3 + 0]);
                g_arr[i] = float(src[src_idx + i * 3 + 1]);
                r_arr[i] = float(src[src_idx + i * 3 + 2]);
            }
            __m256 b_f = _mm256_loadu_ps(b_arr);
            __m256 g_f = _mm256_loadu_ps(g_arr);
            __m256 r_f = _mm256_loadu_ps(r_arr);

            b_f = _mm256_mul_ps(_mm256_sub_ps(b_f, _mm256_set1_ps(mean0)), _mm256_set1_ps(scale0));
            g_f = _mm256_mul_ps(_mm256_sub_ps(g_f, _mm256_set1_ps(mean1)), _mm256_set1_ps(scale1));
            r_f = _mm256_mul_ps(_mm256_sub_ps(r_f, _mm256_set1_ps(mean2)), _mm256_set1_ps(scale2));

            _mm256_storeu_ps(dst + dst_idx0, b_f);
            _mm256_storeu_ps(dst + dst_idx1, g_f);
            _mm256_storeu_ps(dst + dst_idx2, r_f);
        }
        // 处理剩余像素
        for (int x = (src_w & ~7); x < src_w; ++x) {
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
    return Format1_OpenMP_AVX2;
}

int postprocess2_native(
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

    for (int y = pad_top; y < dst_h - pad_bot; ++y) {
        for (int x = pad_lft; x < dst_w - pad_rig; ++x) {
            int src_y = y - pad_top;
            int src_x = x - pad_lft;
            int src_idx = (src_y * src_w + src_x) * 3;
            int dst_idx0 = 0 * size + y * dst_w + x;
            int dst_idx1 = 1 * size + y * dst_w + x;
            int dst_idx2 = 2 * size + y * dst_w + x;
            dst[dst_idx0] = (float(src[src_idx + 0]) - mean0) * scale0;
            dst[dst_idx1] = (float(src[src_idx + 1]) - mean1) * scale1;
            dst[dst_idx2] = (float(src[src_idx + 2]) - mean2) * scale2;
        }
    }
    return Format2_Native;
}

int postprocess2_openmp(
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

    #pragma omp parallel for num_threads(4) schedule(static)
    for (int y = pad_top; y < dst_h - pad_bot; ++y) {
        for (int x = pad_lft; x < dst_w - pad_rig; ++x) {
            int src_y = y - pad_top;
            int src_x = x - pad_lft;
            int src_idx = (src_y * src_w + src_x) * 3;
            int dst_idx0 = 0 * size + y * dst_w + x;
            int dst_idx1 = 1 * size + y * dst_w + x;
            int dst_idx2 = 2 * size + y * dst_w + x;
            dst[dst_idx0] = (float(src[src_idx + 0]) - mean0) * scale0;
            dst[dst_idx1] = (float(src[src_idx + 1]) - mean1) * scale1;
            dst[dst_idx2] = (float(src[src_idx + 2]) - mean2) * scale2;
        }
    }
    return Format2_OpenMP;
}

int postprocess2_openmp_avx2(
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

    const __m256 vmean0 = _mm256_set1_ps(mean0);
    const __m256 vmean1 = _mm256_set1_ps(mean1);
    const __m256 vmean2 = _mm256_set1_ps(mean2);
    const __m256 vscale0 = _mm256_set1_ps(scale0);
    const __m256 vscale1 = _mm256_set1_ps(scale1);
    const __m256 vscale2 = _mm256_set1_ps(scale2);

    #pragma omp parallel for num_threads(4) schedule(static)
    for (int y = pad_top; y < dst_h - pad_bot; ++y) {
        int src_y = y - pad_top;
        for (int x = pad_lft; x + 7 < dst_w - pad_rig; x += 8) {
            int src_x = x - pad_lft;
            float b_arr[8], g_arr[8], r_arr[8];
            int src_idx_base = (src_y * src_w + src_x) * 3;
            for (int i = 0; i < 8; ++i) {
                int src_idx = src_idx_base + i * 3;
                b_arr[i] = float(src[src_idx + 0]);
                g_arr[i] = float(src[src_idx + 1]);
                r_arr[i] = float(src[src_idx + 2]);
            }
            __m256 b_f = _mm256_loadu_ps(b_arr);
            __m256 g_f = _mm256_loadu_ps(g_arr);
            __m256 r_f = _mm256_loadu_ps(r_arr);

            b_f = _mm256_mul_ps(_mm256_sub_ps(b_f, vmean0), vscale0);
            g_f = _mm256_mul_ps(_mm256_sub_ps(g_f, vmean1), vscale1);
            r_f = _mm256_mul_ps(_mm256_sub_ps(r_f, vmean2), vscale2);

            int dst_idx_base = y * dst_w + x;
            _mm256_storeu_ps(dst + 0 * size + dst_idx_base, b_f);
            _mm256_storeu_ps(dst + 1 * size + dst_idx_base, g_f);
            _mm256_storeu_ps(dst + 2 * size + dst_idx_base, r_f);
        }
        // 处理剩余像素
        for (int x = ((dst_w - pad_rig) & ~7); x < dst_w - pad_rig; ++x) {
            int src_x = x - pad_lft;
            int src_idx = (src_y * src_w + src_x) * 3;
            int dst_idx0 = 0 * size + y * dst_w + x;
            int dst_idx1 = 1 * size + y * dst_w + x;
            int dst_idx2 = 2 * size + y * dst_w + x;
            dst[dst_idx0] = (float(src[src_idx + 0]) - mean0) * scale0;
            dst[dst_idx1] = (float(src[src_idx + 1]) - mean1) * scale1;
            dst[dst_idx2] = (float(src[src_idx + 2]) - mean2) * scale2;
        }
    }
    return Format2_OpenMP_AVX2;
}

int postprocess3_native(
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
    return Format3_Native;
}

int postprocess3_openmp(
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
    
    #pragma omp parallel for num_threads(3) schedule(static)
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
    return Format3_OpenMP;
}

int postprocess3_openmp_avx2(
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
    return Format3_OpenMP_AVX2;
}
