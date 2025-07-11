#ifndef __Format__
#define __Format__

#define Format_Error                  0
#define Format1_Native              101
#define Format1_OpenMP              102
#define Format1_OpenMP_AVX2         103
#define Format2_Native              201
#define Format2_OpenMP              202
#define Format2_OpenMP_AVX2         203
#define Format3_Native              301
#define Format3_OpenMP              302
#define Format3_OpenMP_AVX2         303
#define Format_Indexing             501
#define Format_End2End              601


int getVersionOpenMP();

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

#endif