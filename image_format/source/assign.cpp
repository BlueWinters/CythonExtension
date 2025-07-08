
#include <immintrin.h>
#include <omp.h>
#include "assign.h"


void assignValue1(
    float* dst,
    int dst_h,
    int dst_w,
    const float* value
) {
    const int total_pixels = dst_h * dst_w;
    #pragma omp parallel for num_threads(3) schedule(static)
    for (int c = 0; c < 3; ++c) {
        const float val = value[c];
        float* ptr = dst + c * total_pixels;
        const __m256 avx_val = _mm256_set1_ps(val);
        int i = 0;
        for (; i <= total_pixels - 8; i += 8) {
            _mm256_storeu_ps(ptr + i, avx_val);
        }
        for (; i < total_pixels; ++i) {
            ptr[i] = val;
        }
    }
}

void assignValue2(
    float* dst,
    int dst_h,
    int dst_w,
    const float* value
)
{
    const int size = dst_h * dst_w;
    #pragma omp parallel for num_threads(3) schedule(static)
    for (int c = 0; c < 3; ++c) {
        const int pre_size = c * size;
        const float val = value[c];
        const __m256 vval = _mm256_set1_ps(val);
        float* ptr = dst + pre_size;

        // 主循环：一次处理32个元素（4个AVX向量）
        int i = 0;
        // 处理完整32元素块
        for (; i <= size - 32; i += 32) {
            _mm256_store_ps(ptr + i, vval);
            _mm256_store_ps(ptr + i + 8, vval);
            _mm256_store_ps(ptr + i + 16, vval);
            _mm256_store_ps(ptr + i + 24, vval);
        }

        // 处理剩余16个元素（2个向量）
        if (i <= size - 16) {
            _mm256_store_ps(ptr + i, vval);
            _mm256_store_ps(ptr + i + 8, vval);
            i += 16;
        }

        // 处理剩余8个元素（1个向量）
        if (i <= size - 8) {
            _mm256_store_ps(ptr + i, vval);
            i += 8;
        }

        // 处理尾部剩余元素（0-7个）
        // 使用SSE处理4元素块
        if (i < size) {
            const __m128 vval128 = _mm_set1_ps(val);
            // 处理4元素块
            for (; i <= size - 4; i += 4) {
                _mm_store_ps(ptr + i, vval128);
            }
            // 处理最后0-3个元素
            for (; i < size; ++i) {
                ptr[i] = val;
            }
        }
    }
}

void assignValue3(
    float* dst,
    int dst_h,
    int dst_w,
    const float* value
)
{
    const int size = dst_h * dst_w;
    #pragma omp parallel for num_threads(3) schedule(static)
    for (int c = 0; c < 3; ++c) {
        const int pre_size = c * size;
        const float val = value[c];  // 查表获取值
        const __m256 vval = _mm256_set1_ps(val);
        float* ptr = dst + pre_size;

        int i = 0;
        // 主循环：一次处理128个元素（16个AVX向量）
        for (; i <= size - 128; i += 128) {
            // 展开16次向量存储操作
            _mm256_store_ps(ptr + i, vval);
            _mm256_store_ps(ptr + i + 8, vval);
            _mm256_store_ps(ptr + i + 16, vval);
            _mm256_store_ps(ptr + i + 24, vval);
            _mm256_store_ps(ptr + i + 32, vval);
            _mm256_store_ps(ptr + i + 40, vval);
            _mm256_store_ps(ptr + i + 48, vval);
            _mm256_store_ps(ptr + i + 56, vval);
            _mm256_store_ps(ptr + i + 64, vval);
            _mm256_store_ps(ptr + i + 72, vval);
            _mm256_store_ps(ptr + i + 80, vval);
            _mm256_store_ps(ptr + i + 88, vval);
            _mm256_store_ps(ptr + i + 96, vval);
            _mm256_store_ps(ptr + i + 104, vval);
            _mm256_store_ps(ptr + i + 112, vval);
            _mm256_store_ps(ptr + i + 120, vval);
        }

        // 分层处理剩余元素（64, 32, 16, 8元素块）
        // 处理64元素块（8个向量）
        if (i <= size - 64) {
            _mm256_store_ps(ptr + i, vval);
            _mm256_store_ps(ptr + i + 8, vval);
            _mm256_store_ps(ptr + i + 16, vval);
            _mm256_store_ps(ptr + i + 24, vval);
            _mm256_store_ps(ptr + i + 32, vval);
            _mm256_store_ps(ptr + i + 40, vval);
            _mm256_store_ps(ptr + i + 48, vval);
            _mm256_store_ps(ptr + i + 56, vval);
            i += 64;
        }

        // 处理32元素块（4个向量）
        if (i <= size - 32) {
            _mm256_store_ps(ptr + i, vval);
            _mm256_store_ps(ptr + i + 8, vval);
            _mm256_store_ps(ptr + i + 16, vval);
            _mm256_store_ps(ptr + i + 24, vval);
            i += 32;
        }

        // 处理16元素块（2个向量）
        if (i <= size - 16) {
            _mm256_store_ps(ptr + i, vval);
            _mm256_store_ps(ptr + i + 8, vval);
            i += 16;
        }

        // 处理8元素块（1个向量）
        if (i <= size - 8) {
            _mm256_store_ps(ptr + i, vval);
            i += 8;
        }

        // 处理尾部剩余元素（0-7个）
        if (i < size) {
            // 使用SSE处理4元素块
            const __m128 vval128 = _mm_set1_ps(val);
            for (; i <= size - 4; i += 4) {
                _mm_store_ps(ptr + i, vval128);
            }
            // 处理最后0-3个元素
            for (; i < size; ++i) {
                ptr[i] = val;
            }
        }
    }
}

// void assignValue4(
//     float* dst,
//     int dst_h,
//     int dst_w,
//     int dst_c,
//     const float* value
// )
// {
//     const int total_pixels = dst_h * dst_w;
//     const int avx_step = 8;  // AVX2处理8个float

//     #pragma omp parallel for
//     for (int c = 0; c < dst_c; ++c) {
//         const float channel_val = value[c];
//         float* channel_start = dst + c * total_pixels;

//         // AVX2向量化部分
//         const __m256 avx_val = _mm256_set1_ps(channel_val);
//         int i = 0;

//         // 处理对齐内存部分 (64字节对齐)
//         const uintptr_t addr = reinterpret_cast<uintptr_t>(channel_start);
//         const int align_offset = (64 - (addr % 64)) / sizeof(float);
//         const int aligned_end = (total_pixels - align_offset) & ~(avx_step - 1);

//         // 处理前部未对齐部分
//         for (i = 0; i < align_offset && i < total_pixels; ++i) {
//             channel_start[i] = channel_val;
//         }

//         // AVX2向量化主循环
//         #pragma omp simd
//         for (i = align_offset; i < aligned_end; i += avx_step) {
//             _mm256_store_ps(channel_start + i, avx_val);
//         }

//         // 处理后部剩余部分
//         for (i = aligned_end; i < total_pixels; ++i) {
//             channel_start[i] = channel_val;
//         }
//     }
// }