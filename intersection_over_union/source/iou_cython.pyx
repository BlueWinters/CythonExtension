# cython: language_level=3
import numpy as np
cimport numpy as np

np.import_array()  # 初始化 numpy C-API


cdef extern from "iou.h":

    float calculateIOU(
        float x1, float y1, float x2, float y2, 
        float x3, float y3, float x4, float y4,
        bint include_edge);
    float calculateIOU(
        int x1, int y1, int x2, int y2, 
        int x3, int y3, int x4, int y4,
        bint include_edge);
    double calculateIOU(
        double x1, double y1, double x2, double y2,
        double x3, double y3, double x4, double y4,
        bint include_edge);


    void calculateIOUPair_OpenMP(
        const int* boxes1, const int N, 
        const int* boxes2, const int M, 
        float* iou, bint include_edge);
    void calculateIOUPair_OpenMP(
        const float* boxes1, const int N, 
        const float* boxes2, const int M, 
        float* iou, bint include_edge);
    void calculateIOUPair_OpenMP(
        const double* boxes1, const int N, 
        const double* boxes2, const int M, 
        double* iou, bint include_edge);


    void calculateIOUPair_Native(
        const int* boxes1, const int N, 
        const int* boxes2, const int M, 
        float* iou, bint include_edge);
    void calculateIOUPair_Native(
        const float* boxes1, const int N, 
        const float* boxes2, const int M, 
        float* iou, bint include_edge);
    void calculateIOUPair_Native(
        const double* boxes1, const int N, 
        const double* boxes2, const int M, 
        double* iou, bint include_edge);



def calculateIOU_int32(
    np.ndarray[np.int32_t, ndim=2]  boxes1, 
    np.ndarray[np.int32_t, ndim=2]  boxes2,
    bint with_openmp=True,
    bint include_edge=False,
):
    cdef int N = boxes1.shape[0]
    cdef int M = boxes2.shape[0]
    cdef np.ndarray[np.float32_t, ndim=2, mode="c"] iou = np.zeros((N, M), dtype=np.float32)
    cdef np.ndarray[np.int32_t, ndim=2, mode="c"] boxes1_c = np.ascontiguousarray(boxes1)
    cdef np.ndarray[np.int32_t, ndim=2, mode="c"] boxes2_c = np.ascontiguousarray(boxes2)
    cdef int* b1 = <int*>boxes1_c.data
    cdef int* b2 = <int*>boxes2_c.data
    cdef float* iou_ptr = <float*>iou.data
    if with_openmp == True:
        calculateIOUPair_OpenMP(b1, N, b2, M, iou_ptr, include_edge)
    else:
        calculateIOUPair_Native(b1, N, b2, M, iou_ptr, include_edge)
    return iou

def calculateIOU_float(
    np.ndarray[np.float32_t, ndim=2]  boxes1, 
    np.ndarray[np.float32_t, ndim=2]  boxes2,
    bint with_openmp=True,
    bint include_edge=False,
):
    cdef int N = boxes1.shape[0]
    cdef int M = boxes2.shape[0]
    cdef np.ndarray[np.float32_t, ndim=2, mode="c"] iou = np.zeros((N, M), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2, mode="c"] boxes1_c = np.ascontiguousarray(boxes1)
    cdef np.ndarray[np.float32_t, ndim=2, mode="c"] boxes2_c = np.ascontiguousarray(boxes2)
    cdef float* b1 = <float*>boxes1_c.data
    cdef float* b2 = <float*>boxes2_c.data
    cdef float* iou_ptr = <float*>iou.data
    if with_openmp == True:
        calculateIOUPair_OpenMP(b1, N, b2, M, iou_ptr, include_edge)
    else:
        calculateIOUPair_Native(b1, N, b2, M, iou_ptr, include_edge)
    return iou

def calculateIOU_double(
    np.ndarray[np.float64_t, ndim=2]  boxes1, 
    np.ndarray[np.float64_t, ndim=2]  boxes2,
    bint with_openmp=True,
    bint include_edge=False,
):
    cdef int N = boxes1.shape[0]
    cdef int M = boxes2.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2, mode="c"] iou = np.zeros((N, M), dtype=np.double)
    cdef np.ndarray[np.float64_t, ndim=2, mode="c"] boxes1_c = np.ascontiguousarray(boxes1)
    cdef np.ndarray[np.float64_t, ndim=2, mode="c"] boxes2_c = np.ascontiguousarray(boxes2)
    cdef double* b1 = <double*>boxes1_c.data
    cdef double* b2 = <double*>boxes2_c.data
    cdef double* iou_ptr = <double*>iou.data
    if with_openmp == True:
        calculateIOUPair_OpenMP(b1, N, b2, M, iou_ptr, include_edge)
    else:
        calculateIOUPair_Native(b1, N, b2, M, iou_ptr, include_edge)
    return iou


def calculateIOU(
    np.ndarray boxes1,
    np.ndarray boxes2,
    bint with_openmp=True,
    bint include_edge=False,
):
    """
    boxes1: shape=(N,4), dtype=int32/float/double
    boxes2: shape=(M,4), dtype=int32/float/double
    """
    if boxes1.ndim != 2 or boxes2.ndim != 2 or boxes1.shape[1] != 4 or boxes2.shape[1] != 4:
        raise ValueError("boxes1/boxes2 must be (N,4)/(M,4) shape")
    if boxes1.dtype != boxes2.dtype:
        raise TypeError("boxes1 and boxes2 must have the same dtype")

    if boxes1.dtype == np.int32:
        return calculateIOU_int32(boxes1, boxes2, with_openmp, include_edge)
    elif boxes1.dtype == np.float32:
        return calculateIOU_float(boxes1, boxes2, with_openmp, include_edge)
    elif boxes1.dtype == np.double:
        return calculateIOU_double(boxes1, boxes2, with_openmp, include_edge)
    else:
        raise TypeError("Only int32, float32, float64 dtypes are supported")