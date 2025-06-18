
import cv2
import numpy as np
import numba
import time
import iou_cython


def functionWrapper(function, prefix):
    def callFunction(*args, **kwargs):
        beg = time.time()
        num = 100
        for n in range(num):
            output = function(*args, **kwargs)
        end = time.time()
        eclipse = end - beg
        average = eclipse * 1000 *1000 / num
        # print('success call: {}({:.4f} ns)'.format(
        #     prefix, average))
        return output, int(average)
    return callFunction


def calculateIOU_Numpy(box1, box2, include_edge=False) -> float:
    xx1 = max(box1[0], box2[0])
    yy1 = max(box1[1], box2[1])
    xx2 = min(box1[2], box2[2])
    yy2 = min(box1[3], box2[3])
    value = 1 if include_edge else 0
    inter_area = (max(0, xx2 - xx1 + value) * max(0, yy2 - yy1 + value))
    area_a = (box1[2] - box1[0] + value) * (box1[3] - box1[1] + value)
    area_b = (box2[2] - box2[0] + value) * (box2[3] - box2[1] + value)
    if area_a == 0 or area_b == 0:
        return float(0.)
    union_area = area_a + area_b - inter_area
    return float(inter_area / union_area if union_area > 0. else 0.)


def calculateIOU_NumpyBatchFor(boxes1, boxes2, include_edge=False):
    iou = np.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=np.float32)
    for i in range(boxes1.shape[0]):
        for j in range(boxes2.shape[0]):
            iou[i, j] = calculateIOU_Numpy(boxes1[i], boxes2[j], include_edge)
    return iou


def calculateIOU_NumpyBatchMatrix(boxes1, boxes2, include_edge=False):
    value = 1 if include_edge else 0
    boxes1 = np.asarray(boxes1, dtype=np.float32)
    boxes2 = np.asarray(boxes2, dtype=np.float32)
    M = boxes1.shape[0]
    N = boxes2.shape[0]
    x_lft = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    y_top = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    x_rig = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    y_bot = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])
    inter_w = np.clip(x_rig - x_lft + value, a_min=0, a_max=None)
    inter_h = np.clip(y_bot - y_top + value, a_min=0, a_max=None)
    inter_area = inter_w * inter_h
    area1 = (boxes1[:, 2] - boxes1[:, 0] + value) * (boxes1[:, 3] - boxes1[:, 1] + value)  # (M,)
    area2 = (boxes2[:, 2] - boxes2[:, 0] + value) * (boxes2[:, 3] - boxes2[:, 1] + value)  # (N,)
    union_area = area1[:, None] + area2[None, :] - inter_area
    iou = np.where(union_area > 0, inter_area / union_area, 0.0)
    return iou


@numba.jit(nopython=True, nogil=True, parallel=True)
def calculateIOU_NumpyBatchNumba(boxes1, boxes2, include_edge=False):
    value = 1 if include_edge else 0
    iou = np.zeros(shape=(boxes1.shape[0], boxes2.shape[0]), dtype=np.float32)
    for i in range(boxes1.shape[0]):
        for j in range(boxes2.shape[0]):
            b1 = boxes1[i]
            b2 = boxes2[j]
            xx1 = max(b1[0], b2[0])
            yy1 = max(b1[1], b2[1])
            xx2 = min(b1[2], b2[2])
            yy2 = min(b1[3], b2[3])
            inter_area = (max(0, xx2 - xx1 + value) * max(0, yy2 - yy1 + value))
            area_a = (b1[2] - b1[0] + value) * (b1[3] - b1[1] + value)
            area_b = (b2[2] - b2[0] + value) * (b2[3] - b2[1] + value)
            if area_a == 0 or area_b == 0:
                continue
            union_area = area_a + area_b - inter_area
            iou[i, j] = inter_area / union_area if union_area > 0. else 0.
    return iou

def makeBatchBoxes(base):
    box1 = np.array([10, 10, 20, 20], dtype=np.float32)
    box2 = np.array([15, 15, 25, 25], dtype=np.float32)
    box3 = np.array([ 5,  5, 40, 40], dtype=np.float32)
    boxes = np.concatenate([
        np.tile(box1, (5*base, 1)), 
        np.tile(box2, (2*base, 1)), 
        np.tile(box3, (3*base, 1))], axis=0)
    boxes = np.ascontiguousarray(boxes)
    return boxes

def benchmark(base_list):
    avearage_time = []

    # num_boxes = []
    # avearage_time = []
    # for base in base_list:
    #     b = makeBatchBoxes(base)
    #     num_boxes.append(b.shape[0])
    #     avearage_time.append(functionWrapper(calculateIOU_NumpyBatchFor, 'numpy_batch_for')(b, b, False)[1])
    # print(num_boxes)
    # print(avearage_time)

    # num_boxes = []
    # avearage_time = []
    # for base in base_list:
    #     b = makeBatchBoxes(base)
    #     num_boxes.append(b.shape[0])
    #     avearage_time.append(functionWrapper(calculateIOU_NumpyBatchMatrix, 'numpy_batch_matrix')(b, b, False)[1])
    # print(num_boxes)
    # print(avearage_time)

    num_boxes = []
    avearage_time = []
    for base in base_list:
        b = makeBatchBoxes(base)
        num_boxes.append(b.shape[0])
        avearage_time.append(functionWrapper(calculateIOU_NumpyBatchNumba, 'numpy_batch_numba')(b, b, False)[1])
    print(num_boxes)
    print(avearage_time)

    # num_boxes = []
    # avearage_time = []
    # for base in base_list:
    #     b = makeBatchBoxes(base)
    #     num_boxes.append(b.shape[0])
    #     avearage_time.append(functionWrapper(iou_cython.calculateIOU_float, 'cpp_native')(b, b, False)[1])
    # print(num_boxes)
    # print(avearage_time)

    # num_boxes = []
    # avearage_time = []
    # for base in base_list:
    #     b = makeBatchBoxes(base)
    #     num_boxes.append(b.shape[0])
    #     avearage_time.append(functionWrapper(iou_cython.calculateIOU_float, 'cpp_openmp')(b, b, True)[1])
    # print(num_boxes)
    # print(avearage_time)


if __name__ == '__main__':
    np.set_printoptions(precision=4)
    benchmark([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50])