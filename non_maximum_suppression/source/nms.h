#ifndef __NMS__
#define __NMS__


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


#endif
