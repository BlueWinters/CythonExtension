# distutils: language = c++
# cython: language_level=3

import numpy as np
cimport numpy as np

np.import_array()  # 初始化 numpy C-API

cdef extern from "nms.h":
    void getKeepIndex(
        const float* in_scores,
        const float* in_boxes,
        int n_input,
        int* keep_indices,
        int& n_keep,
        float threshold
    );

    void getResult(
        const float* in_scores,
        const float* in_boxes,
        const int n_keep,
        const int* keep_indices,
        float*& out_scores,
        float*& out_boxes
    );


"""
interface for python
"""
def doNMS(
    np.ndarray[np.float32_t, ndim=1]  scores,
    np.ndarray[np.float32_t, ndim=2]  boxes,
    float threshold,
):
    cdef int num_scores = scores.shape[0]
    cdef int num_boxes = boxes.shape[0]

    if num_scores != num_boxes:
        raise ValueError("scores/boxes should have the length on dim 0")
	
    cdef np.ndarray[np.float32_t, ndim=1, mode="c"] scores_c = np.ascontiguousarray(scores)
    cdef np.ndarray[np.float32_t, ndim=2, mode="c"] boxes_c = np.ascontiguousarray(boxes)

    cdef np.ndarray[np.int32_t, ndim=1, mode="c"] keep_indices = np.zeros((num_scores,), dtype=np.int32)
    cdef int n_keep = 0
    cdef float* scores_ptr = <float*>scores_c.data
    cdef float* boxes_ptr = <float*>boxes_c.data
    cdef int* keep_indices_ptr = <int*>keep_indices.data
    getKeepIndex(scores_ptr, boxes_ptr, num_scores, keep_indices_ptr, n_keep, threshold)

    cdef np.ndarray[np.float32_t, ndim=1, mode="c"] out_scores = np.empty((n_keep,), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2, mode="c"] out_boxes = np.empty((n_keep, 4), dtype=np.float32)
    cdef float* out_scores_ptr = <float*>out_scores.data
    cdef float* out_boxes_ptr = <float*>out_boxes.data
    getResult(scores_ptr, boxes_ptr, n_keep, keep_indices_ptr, out_scores_ptr, out_boxes_ptr)
    return out_scores, out_boxes


