#ifndef __Format__
#define __Format__

#define Format_Error               0
#define Format_Native              1
#define Format_OpenMP              2
#define Format_OpenMP_AVX2         3


bool preprocess(
    int src_h, 
    int src_w, 
    int dst_h, 
    int dst_w, 
    int& rsz_h,
    int& rsz_w, 
    int& pad_lft,
    int& pad_rig,
    int& pad_top,
    int& pad_bot
);

int postprocess_native(
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

int postprocess_openmp1(
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

int postprocess_openmp2(
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

int postprocess_openmp3(
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

int postprocess_full(
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

#endif