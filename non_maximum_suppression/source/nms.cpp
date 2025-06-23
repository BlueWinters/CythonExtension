
#include <algorithm>
#include "nms.h"


void getKeepIndex(
    const float* in_scores,
    const float* in_boxes,
    int n_input,
    int* keep_indices,
    int& n_keep,
    float iou_threshold
) 
{
    n_keep = 0;
    if (in_scores == nullptr || in_boxes == nullptr || 
        n_input == 0 || keep_indices == 0) {
        return;
    }

    int* order = new int[n_input];
    float* areas = new float[n_input];
    for (int i = 0; i < n_input; ++i) {
        order[i] = i;
        areas[i] = (in_boxes[i*4+2] - in_boxes[i*4+0] + 1) * (in_boxes[i*4+3] - in_boxes[i*4+1] + 1);
    }

    std::sort(order, order + n_input, [&](int a, int b) {
        if (in_scores[a] != in_scores[b])
            return in_scores[a] > in_scores[b];
        return areas[a] > areas[b];
    });

    bool* suppressed = new bool[n_input];
    std::memset(suppressed, 0, n_input * sizeof(bool));

    for (int _i = 0; _i < n_input; ++_i) {
        int i = order[_i];
        if (suppressed[i]) continue;
        keep_indices[n_keep++] = i;
        for (int _j = _i + 1; _j < n_input; ++_j) {
            int j = order[_j];
            if (suppressed[j]) continue;
            float xx1 = std::max(in_boxes[i*4+0], in_boxes[j*4+0]);
            float yy1 = std::max(in_boxes[i*4+1], in_boxes[j*4+1]);
            float xx2 = std::min(in_boxes[i*4+2], in_boxes[j*4+2]);
            float yy2 = std::min(in_boxes[i*4+3], in_boxes[j*4+3]);
            float w = std::max(0.0f, xx2 - xx1 + 1);
            float h = std::max(0.0f, yy2 - yy1 + 1);
            float inter = w * h;
            float ovr = inter / (areas[i] + areas[j] - inter);
            if (ovr > iou_threshold) suppressed[j] = true;
        }
    }
    delete[] order;
    delete[] areas;
    delete[] suppressed;
}

void getResult(
    const float* in_scores,
    const float* in_boxes,
    const int n_keep,
    const int* keep_indices,
    float*& out_scores,
    float*& out_boxes
)
{
    for (int k = 0; k < n_keep; ++k) {
        int idx = keep_indices[k];
        out_scores[k] = in_scores[idx];
        std::memcpy(out_boxes + k*4, in_boxes + idx*4, 4 * sizeof(float));
    }
}
