
#include <type_traits>
#include "iou.h"


#ifndef StdMax
#define StdMax(a,b)  (((a) > (b)) ? (a) : (b))
#endif
#ifndef StdMin
#define StdMin(a,b)  (((a) < (b)) ? (a) : (b))
#endif


#if __cplusplus >= 201703L
// do not need to specialize for int, as C++17 allows constexpr if
#pragma message("\033[31m" "C++17 or later detected, using constexpr if for type specialization." "\033[0m")
#else
inline float divide_inline(float* inter_area, float* union_area) {
    return (*inter_area) / (*union_area);
}

inline double divide_inline(double* inter_area, double* union_area) {
    return (*inter_area) / (*union_area);
}

inline float divide_inline(int* inter_area, int* union_area) {
    return static_cast<float>(static_cast<float>(*inter_area) / static_cast<float>(*union_area));
}
#endif



template<typename dtype>
void calculateIOU_Template(
    dtype x1, dtype y1, dtype x2, dtype y2,
    dtype x3, dtype y3, dtype x4, dtype y4,
    dtype& inter_area, dtype& union_area,
    bool include_edge)
{
    dtype x_lft = StdMax(x1, x3);
    dtype y_top = StdMax(y1, y3);
    dtype x_rig = StdMin(x2, x4);
    dtype y_bot = StdMin(y2, y4);
    dtype value = static_cast<dtype>(include_edge == true ? 1 : 0);
    dtype inter_width  = x_rig - x_lft + value;
    dtype inter_height = y_bot - y_top + value;

    if (inter_width <= 0 || inter_height <= 0)
        return;

    inter_area = inter_width * inter_height;
    dtype area_a = (x2 - x1 + value) * (y2 - y1 + value);
    dtype area_b = (x4 - x3 + value) * (y4 - y3 + value);
    union_area = area_a + area_b - inter_area;

//     if (union_area <= 0)
//         return 0.0f;

// #if __cplusplus >= 201703L
//     if constexpr (std::is_same_v<dtype, int>) {
//         return static_cast<dtype>(static_cast<float>(inter_area) / static_cast<float>(union_area));
//     } else {
//         return inter_area / union_area;
//     }
// #else
//     return divide_inline(&inter_area, &union_area);
// #endif
}



float calculateIOU(
    int x1, int y1, int x2, int y2,
    int x3, int y3, int x4, int y4,
    bool include_edge) 
{
    int inter_area = 0, union_area = 0;
    calculateIOU_Template<int>(x1, y1, x2, y2, x3, y3, x4, y4, inter_area, union_area, include_edge);
    if (union_area <= 0)
        return 0.f;
    return static_cast<float>(inter_area) / static_cast<float>(union_area);
}

float calculateIOU(
    float x1, float y1, float x2, float y2,
    float x3, float y3, float x4, float y4,
    bool include_edge) 
{
    float inter_area = 0.f, union_area = 0.f;
    calculateIOU_Template<float>(x1, y1, x2, y2, x3, y3, x4, y4, inter_area, union_area, include_edge);
    if (union_area <= 0.f)
        return static_cast<float>(0.f);
    return static_cast<float>(inter_area) / static_cast<float>(union_area);
}

double calculateIOU(
    double x1, double y1, double x2, double y2,
    double x3, double y3, double x4, double y4,
    bool include_edge) 
{
    double inter_area = 0, union_area = 0;
    calculateIOU_Template<double>(x1, y1, x2, y2, x3, y3, x4, y4, inter_area, union_area, include_edge);
    if (union_area <= 0.f)
        return static_cast<double>(0.f);
    return static_cast<double>(inter_area) / static_cast<double>(union_area);
}


template<typename dtype, typename dtype_iou>
void calculateIOUPair_Native_Template(
    const dtype* boxes1, const int N, 
    const dtype* boxes2, const int M, 
    dtype_iou* iou, bool include_edge)
{
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            const dtype* box1 = boxes1 + i * 4;
            const dtype* box2 = boxes2 + j * 4;
            iou[i * M + j] = static_cast<dtype_iou>(
                calculateIOU(
                    box1[0], box1[1], box1[2], box1[3],
                    box2[0], box2[1], box2[2], box2[3],
                    include_edge
                )
            );
        }
    }
}

template<typename dtype, typename dtype_iou>
void calculateIOUPair_OpenMP_Template(
    const dtype* boxes1, const int N, 
    const dtype* boxes2, const int M, 
    dtype_iou* iou, bool include_edge)
{
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            const dtype* box1 = boxes1 + i * 4;
            const dtype* box2 = boxes2 + j * 4;
            iou[i * M + j] = static_cast<dtype_iou>(
                calculateIOU(
                    box1[0], box1[1], box1[2], box1[3],
                    box2[0], box2[1], box2[2], box2[3],
                    include_edge
                )
            );
        }
    }
}

void calculateIOUPair_OpenMP(
    const int* boxes1, const int N, 
    const int* boxes2, const int M, 
    float* iou, bool include_edge)
{
    calculateIOUPair_OpenMP_Template<int, float>(boxes1, N, boxes2, M, iou, include_edge);
}

void calculateIOUPair_OpenMP(
    const float* boxes1, const int N, 
    const float* boxes2, const int M, 
    float* iou, bool include_edge)
{
    calculateIOUPair_OpenMP_Template<float, float>(boxes1, N, boxes2, M, iou, include_edge);
}

void calculateIOUPair_OpenMP(
    const double* boxes1, const int N, 
    const double* boxes2, const int M, 
    double* iou, bool include_edge)
{
    calculateIOUPair_OpenMP_Template<double, double>(boxes1, N, boxes2, M, iou, include_edge);
}



void calculateIOUPair_Native(
    const int* boxes1, const int N, 
    const int* boxes2, const int M, 
    float* iou, bool include_edge)
{
    calculateIOUPair_Native_Template<int, float>(boxes1, N, boxes2, M, iou, include_edge);
}

void calculateIOUPair_Native(
    const float* boxes1, const int N, 
    const float* boxes2, const int M, 
    float* iou, bool include_edge)
{
    calculateIOUPair_Native_Template<float, float>(boxes1, N, boxes2, M, iou, include_edge);
}

void calculateIOUPair_Native(
    const double* boxes1, const int N, 
    const double* boxes2, const int M, 
    double* iou, bool include_edge)
{
    calculateIOUPair_Native_Template<double, double>(boxes1, N, boxes2, M, iou, include_edge);
}

