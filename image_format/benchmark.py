
import cv2
import numpy as np
import time
import format_cython

np.set_printoptions(precision=1)


def functionWrapper(function, prefix):
    def callFunction(images, *args, **kwargs):
        time_list = []
        output = function(np.zeros_like(images[0], dtype=np.uint8), *args, **kwargs)
        for n, image in enumerate(images):
            beg = time.perf_counter()
            function(image, *args, **kwargs)
            end = time.perf_counter()
            time_list.append(end - beg)
        time_array = np.array(time_list, dtype=np.float32) * 1000 * 1000
        t_min, t_max = np.min(time_array), np.max(time_array)
        t_avg, t_var = np.mean(time_array), np.std(time_array)
        print('success call: {:<48}({:12.1f} ns, {:12.1f} ns, {:12.1f} {} {:12.1f} ns)'.format(
            prefix, t_min, t_max, t_avg, chr(177), t_var))
        return output
    return callFunction


def check():
    src = np.ascontiguousarray(cv2.imread('asset/input.png'))
    print(src.shape)
    padding_value = 0
    mean_value = (104, 117, 123)
    scale_value = (1., 1., 1.)

    fmt1, pad1 = formatWithNumpy(src, 500, 320, padding_value, mean_value, scale_value)
    fmt2, pad2, flag2 = format_cython.formatImage(src, 500, 320, padding_value, *mean_value, *scale_value, 'openmp')
    print('check openmp: ', flag2, pad1, pad2, np.abs(fmt1 - fmt2).sum())
    fmt3, pad3, flag3 = format_cython.formatImage(src, 500, 320, padding_value, *mean_value, *scale_value, 'openmp_avx2')
    print('check openmp_avx2: ', flag3, pad1, pad3, np.abs(fmt1 - fmt3).sum())


def formatWithNumpy(bgr, dst_h, dst_w, padding_value, mean, scale):
    src_h, src_w, _ = bgr.shape
    src_ratio = float(src_h / src_w)
    dst_ratio = float(dst_h / dst_w)
    if src_ratio > dst_ratio:
        rsz_h, rsz_w = dst_h, int(round(float(src_w / src_h) * dst_h))
        resized = cv2.resize(bgr, (max(1, rsz_w), max(1, rsz_h)))
        lp = (dst_w - rsz_w) // 2
        rp = dst_w - rsz_w - lp
        resized = np.pad(resized, ((0, 0), (lp, rp), (0, 0)), constant_values=padding_value, mode='constant')
        padding = (0, 0, lp, rp)
    else:
        rsz_h, rsz_w = int(round(float(src_h / src_w) * dst_w)), dst_w
        resized = cv2.resize(bgr, (max(1, rsz_w), max(1, rsz_h)))
        tp = (dst_h - rsz_h) // 2
        bp = dst_h - rsz_h - tp
        resized = np.pad(resized, ((tp, bp), (0, 0), (0, 0)), constant_values=padding_value, mode='constant')
        padding = (tp, bp, 0, 0)
    batch_image = cv2.dnn.blobFromImage(resized, scalefactor=1., mean=mean, swapRB=False)
    return batch_image, padding


def performanceOnSingle(dst_h, dst_w):
    src = [np.random.randint(0, 255, (512, 512, 3)).astype(np.uint8) for n in range(50)]
    padding = 0
    mean = (104, 117, 123)
    scale = (1., 1., 1.)

    print('---------------------------------')
    functionWrapper(formatWithNumpy, f'python-({dst_h}, {dst_w})')(src, dst_h, dst_w, padding, mean, scale)
    functionWrapper(format_cython.formatImage, f'cython-openmp-({dst_h}, {dst_w})')(src, dst_h, dst_w, padding, *mean, *scale, 'openmp')
    functionWrapper(format_cython.formatImage, f'cython-openmp_avx2-({dst_h}, {dst_w})')(src, dst_h, dst_w, padding, *mean, *scale, 'openmp_avx2')


if __name__ == '__main__':
    check()
    performanceOnSingle(256, 256)
    performanceOnSingle(512, 512)
    performanceOnSingle(768, 768)
    performanceOnSingle(1024, 1024)

