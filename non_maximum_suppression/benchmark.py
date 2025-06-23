
import cv2
import numpy as np
import time
import pickle
import nms_cython


def functionWrapper(function, prefix):
    def callFunction(*args, **kwargs):
        beg = time.time()
        num = 1000
        for n in range(num):
            output = function(*args, **kwargs)
        end = time.time()
        eclipse = end - beg
        average = eclipse * 1000 * 1000 / num
        print('success call: {}({:.4f} ns)'.format(
            prefix, average))
        return int(average)
    return callFunction


def visual_targets_cv2(bgr, scores, boxes, options=(True, True)):
    visual_score, visual_points = options
    for s, b in zip(scores, boxes):
        # bounding box
        point1 = int(b[0]), int(b[1])
        point2 = int(b[2]), int(b[3])
        cv2.rectangle(bgr, point1, point2, (255, 0, 0), 1)
    return bgr


def doNMS_py(scores, boxes, nms_threshold):
    detections = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)

    x1 = detections[:, 0]
    y1 = detections[:, 1]
    x2 = detections[:, 2]
    y2 = detections[:, 3]
    scores = detections[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        index = np.where(ovr <= nms_threshold)[0]
        order = order[index + 1]
    return scores[keep], boxes[keep]


def benchmark_check():
    bgr = cv2.imread('asset/14.jpg')

    data = pickle.load(open('asset/14.pkl', 'rb'))
    scores, boxes = data['scores'], data['boxes']
    print('input-score: ', scores.shape, scores.dtype)
    print('input-boxes: ', boxes.shape, boxes.dtype)
    canvas = visual_targets_cv2(np.copy(bgr), scores, boxes)
    cv2.imwrite('asset/output-origin.png', canvas)

    scores, boxes = doNMS_py(scores, boxes, 0.4)
    print('output-score: ', scores.shape, scores.dtype)
    print('output-boxes: ', boxes.shape, boxes.dtype)
    canvas = visual_targets_cv2(np.copy(bgr), scores, boxes)
    cv2.imwrite('asset/output-python.png', canvas)

    scores, boxes = nms_cython.doNMS(scores, boxes, 0.4)
    print('output-score: ', scores.shape, scores.dtype)
    print('output-boxes: ', boxes.shape, boxes.dtype)
    canvas = visual_targets_cv2(np.copy(bgr), scores, boxes)
    cv2.imwrite('asset/output-cython.png', canvas)


def benchmark_time(path_list):
    nms_threshold = 0.4

    proposals_len = []
    for path in path_list:
        data = pickle.load(open(path, 'rb'))
        scores, boxes = data['scores'], data['boxes']
        proposals_len.append(len(scores))

    average_time_numpy = []
    for path in path_list:
        data = pickle.load(open(path, 'rb'))
        scores, boxes = data['scores'], data['boxes']
        average_time_numpy.append(functionWrapper(doNMS_py, 'nms_numpy')(scores, boxes, nms_threshold))
    
    average_time_cython = []
    for path in path_list:
        data = pickle.load(open(path, 'rb'))
        scores, boxes = data['scores'], data['boxes']
        average_time_cython.append(functionWrapper(nms_cython.doNMS, 'nms_cython')(scores, boxes, nms_threshold))

    proposals_len = np.array(proposals_len, dtype=np.int32)
    average_time_numpy = np.array(average_time_numpy, dtype=np.int32)
    average_time_cython = np.array(average_time_cython, dtype=np.int32)
    index = np.argsort(proposals_len)
    proposals_len = proposals_len[index]
    average_time_numpy = average_time_numpy[index]
    average_time_cython = average_time_cython[index]
    ratio = average_time_numpy.astype(np.float32) / average_time_cython.astype(np.float32)
    print(proposals_len)
    print(average_time_numpy)
    print(average_time_cython)
    print(ratio)


if __name__ == '__main__':
    np.set_printoptions(precision=1)
    benchmark_check()
    # benchmark_time(['asset/{}.pkl'.format(n) for n in range(1, 15)])
