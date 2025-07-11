# distutils: language = c++
# cython: language_level=3

import numpy as np
import cv2
from openmp cimport omp_get_thread_num, omp_get_num_threads
cimport numpy as np
cimport openmp


np.import_array()  # 初始化 numpy C-API

cdef extern from "format.h":
    int Format_Error
    int Format1_Native
    int Format1_OpenMP
    int Format1_OpenMP_AVX2
    int Format2_Native
    int Format2_OpenMP
    int Format2_OpenMP_AVX2
    int Format3_Native
    int Format3_OpenMP
    int Format3_OpenMP_AVX2
    int Format_Indexing
    int Format_End2End

    int getVersionOpenMP();

    bint preprocess(
        int src_h, 
        int src_w, 
        int dst_h, 
        int dst_w, 
        int& rsz_h,
        int& rsz_w, 
        int& pad_lft,
        int& pad_rig,
        int& pad_top,
        int& pad_bot,
    );

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
    );

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
    );

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
    );

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
    );

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
    );

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
    );

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
    );

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
    );

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
    );

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
    );

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
    );

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
    );

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
    );

"""
"""
def testOpenMP():
    cdef int num_threads = 0
    # 使用OpenMP函数
    num_threads = omp_get_num_threads()
    return num_threads > 0

def testOpenMPVersion():
    return getVersionOpenMP()

def getAlignedArray(shape, dtype, alignment=64):
    n_bytes = np.prod(shape) * np.dtype(dtype).itemsize
    buffer = np.empty(n_bytes + alignment, dtype=np.uint8)
    address = buffer.ctypes.data
    offset = (alignment - (address % alignment)) % alignment
    aligned_buffer = buffer[offset:offset + n_bytes]
    return aligned_buffer.view(dtype).reshape(shape)

"""
interface for python
"""
def formatImage(
    np.ndarray[np.uint8_t, ndim=3]  src,
    int dst_h,
    int dst_w,
    int padding_value,
    float mean0,
    float mean1,
    float mean2,
    float scale0,
    float scale1,
    float scale2,
    str parallel = 'openmp',
):
    if not (dst_h > 0 and dst_w > 0):
        raise ValueError("target shape error, dst_h/dst_w > 0")
    cdef int src_h = src.shape[0]
    cdef int src_w = src.shape[1]
    if not (src_h > 0 and src_w > 0):
        raise ValueError("source shape error, src_h/src_w > 0")

    cdef int rsz_h = 0
    cdef int rsz_w = 0
    cdef int pad_lft = 0
    cdef int pad_rig = 0
    cdef int pad_top = 0
    cdef int pad_bot = 0
    preprocess(
        src_h, src_w,
        dst_h, dst_w,
        rsz_h, rsz_w,
        pad_lft, pad_rig,
        pad_top, pad_bot,
    )

    cdef np.ndarray[np.float32_t, ndim=4, mode="c"] bgr_fmt = np.zeros((1, 3, dst_h, dst_w), dtype=np.float32)
    # cdef np.ndarray[np.float32_t, ndim=4, mode="c"] bgr_fmt = getAlignedArray((1, 3, dst_h, dst_w), np.float32, 32)
    if parallel == 'openmp_end2end':
        flag = postprocess_end2end_v2(
            <const unsigned char*>src.data,
            src_h, src_w,
            rsz_h, rsz_w,
            pad_lft, pad_top,
            pad_rig, pad_bot,
            padding_value,
            mean0, mean1, mean2,
            scale0, scale1, scale2,
            <float*>bgr_fmt.data,
            dst_h, dst_w,
        )
        return bgr_fmt, (pad_top, pad_bot, pad_lft, pad_rig), flag
    if parallel == 'openmp_indexing':
        flag = postprocess_indexing(
            <const unsigned char*>src.data,
            src_h, src_w,
            rsz_h, rsz_w,
            pad_lft, pad_top,
            pad_rig, pad_bot,
            padding_value,
            mean0, mean1, mean2,
            scale0, scale1, scale2,
            <float*>bgr_fmt.data,
            dst_h, dst_w,
        )
        return bgr_fmt, (pad_top, pad_bot, pad_lft, pad_rig), flag

    cdef np.ndarray[np.uint8_t, ndim=3, mode="c"] resized = cv2.resize(src, (rsz_w, rsz_h))
    if parallel == 'native1':
        flag = postprocess1_native(
            <const unsigned char*>resized.data,
            rsz_h, rsz_w,
            pad_lft, pad_top,
            pad_rig, pad_bot,
            mean0, mean1, mean2,
            scale0, scale1, scale2,
            padding_value,
            <float*>bgr_fmt.data,
            dst_h, dst_w,
        )
        return bgr_fmt, (pad_top, pad_bot, pad_lft, pad_rig), flag
    if parallel == 'native1_openmp':
        flag = postprocess1_openmp(
            <const unsigned char*>resized.data,
            rsz_h, rsz_w,
            pad_lft, pad_top,
            pad_rig, pad_bot,
            mean0, mean1, mean2,
            scale0, scale1, scale2,
            padding_value,
            <float*>bgr_fmt.data,
            dst_h, dst_w,
        )
        return bgr_fmt, (pad_top, pad_bot, pad_lft, pad_rig), flag
    if parallel == 'native1_openmp_sse2':
        flag = postprocess1_openmp_sse2(
            <const unsigned char*>resized.data,
            rsz_h, rsz_w,
            pad_lft, pad_top,
            pad_rig, pad_bot,
            mean0, mean1, mean2,
            scale0, scale1, scale2,
            padding_value,
            <float*>bgr_fmt.data,
            dst_h, dst_w,
        )
        return bgr_fmt, (pad_top, pad_bot, pad_lft, pad_rig), flag
    if parallel == 'native1_openmp_avx2':
        flag = postprocess1_openmp_avx2(
            <const unsigned char*>resized.data,
            rsz_h, rsz_w,
            pad_lft, pad_top,
            pad_rig, pad_bot,
            mean0, mean1, mean2,
            scale0, scale1, scale2,
            padding_value,
            <float*>bgr_fmt.data,
            dst_h, dst_w,
        )
        return bgr_fmt, (pad_top, pad_bot, pad_lft, pad_rig), flag
    if parallel == 'native2':
        flag = postprocess2_native(
            <const unsigned char*>resized.data,
            rsz_h, rsz_w,
            pad_lft, pad_top,
            pad_rig, pad_bot,
            mean0, mean1, mean2,
            scale0, scale1, scale2,
            padding_value,
            <float*>bgr_fmt.data,
            dst_h, dst_w,
        )
        return bgr_fmt, (pad_top, pad_bot, pad_lft, pad_rig), flag
    if parallel == 'native2_openmp':
        flag = postprocess2_openmp(
            <const unsigned char*>resized.data,
            rsz_h, rsz_w,
            pad_lft, pad_top,
            pad_rig, pad_bot,
            mean0, mean1, mean2,
            scale0, scale1, scale2,
            padding_value,
            <float*>bgr_fmt.data,
            dst_h, dst_w,
        )
        return bgr_fmt, (pad_top, pad_bot, pad_lft, pad_rig), flag
    if parallel == 'native2_openmp_avx2':
        flag = postprocess2_openmp_avx2(
            <const unsigned char*>resized.data,
            rsz_h, rsz_w,
            pad_lft, pad_top,
            pad_rig, pad_bot,
            mean0, mean1, mean2,
            scale0, scale1, scale2,
            padding_value,
            <float*>bgr_fmt.data,
            dst_h, dst_w,
        )
        return bgr_fmt, (pad_top, pad_bot, pad_lft, pad_rig), flag
    if parallel == 'native3':
        flag = postprocess3_native(
            <const unsigned char*>resized.data,
            rsz_h, rsz_w,
            pad_lft, pad_top,
            pad_rig, pad_bot,
            mean0, mean1, mean2,
            scale0, scale1, scale2,
            padding_value,
            <float*>bgr_fmt.data,
            dst_h, dst_w,
        )
        return bgr_fmt, (pad_top, pad_bot, pad_lft, pad_rig), flag
    if parallel == 'native3_openmp':
        flag = postprocess3_openmp(
            <const unsigned char*>resized.data,
            rsz_h, rsz_w,
            pad_lft, pad_top,
            pad_rig, pad_bot,
            mean0, mean1, mean2,
            scale0, scale1, scale2,
            padding_value,
            <float*>bgr_fmt.data,
            dst_h, dst_w,
        )
        return bgr_fmt, (pad_top, pad_bot, pad_lft, pad_rig), flag
    if parallel == 'native3_openmp_avx2':
        flag = postprocess3_openmp_avx2(
            <const unsigned char*>resized.data,
            rsz_h, rsz_w,
            pad_lft, pad_top,
            pad_rig, pad_bot,
            mean0, mean1, mean2,
            scale0, scale1, scale2,
            padding_value,
            <float*>bgr_fmt.data,
            dst_h, dst_w,
        )
        return bgr_fmt, (pad_top, pad_bot, pad_lft, pad_rig), flag
    raise ValueError('unknown parallel method: {}'.format(parallel))


Result_Format_Error = Format_Error
Result_Format1_Native = Format1_Native
Result_Format1_OpenMP = Format1_OpenMP
Result_Format1_OpenMP_AVX2 = Format1_OpenMP_AVX2
Result_Format2_Native = Format2_Native
Result_Format2_OpenMP = Format2_OpenMP
Result_Format2_OpenMP_AVX2 = Format2_OpenMP_AVX2
Result_Format3_Native = Format3_Native
Result_Format3_OpenMP = Format3_OpenMP
Result_Format3_OpenMP_AVX2 = Format3_OpenMP_AVX2
Result_Format_Indexing = Format_Indexing
Result_Format_End2End = Format_End2End
