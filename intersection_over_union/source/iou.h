#ifndef __IOU__
#define __IOU__


float calculateIOU(
    int x1, int y1, int x2, int y2, 
    int x3, int y3, int x4, int y4,
    bool include_edge);
float calculateIOU(
    float x1, float y1, float x2, float y2, 
    float x3, float y3, float x4, float y4,
    bool include_edge);
double calculateIOU(
    double x1, double y1, double x2, double y2,
    double x3, double y3, double x4, double y4,
    bool include_edge);


void calculateIOUPair_OpenMP(
    const int* boxes1, const int N, 
    const int* boxes2, const int M, 
    float* iou, bool include_edge);
void calculateIOUPair_OpenMP(
    const float* boxes1, const int N, 
    const float* boxes2, const int M, 
    float* iou, bool include_edge);
void calculateIOUPair_OpenMP(
    const double* boxes1, const int N, 
    const double* boxes2, const int M, 
    double* iou, bool include_edge);


void calculateIOUPair_Native(
    const int* boxes1, const int N, 
    const int* boxes2, const int M, 
    float* iou, bool include_edge);
void calculateIOUPair_Native(
    const float* boxes1, const int N, 
    const float* boxes2, const int M, 
    float* iou, bool include_edge);
void calculateIOUPair_Native(
    const double* boxes1, const int N, 
    const double* boxes2, const int M, 
    double* iou, bool include_edge);
    
#endif