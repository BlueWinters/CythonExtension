import cv2
import numpy as np
import time
import pickle
import format_cython

np.set_printoptions(precision=1)


def functionWrapper(function, prefix, path_save):
    def callFunction(num, *args, **kwargs):
        time_list = []
        output = function(np.zeros(shape=(512, 512, 3), dtype=np.uint8), *args, **kwargs)
        # images = [np.random.randint(0, 255, (960, 540, 3)).astype(np.uint8) for _ in range(num)]
        for n in range(num):
            image = np.random.randint(0, 255, (960, 540, 3)).astype(np.uint8)
            # image = images[n]
            beg = time.perf_counter()
            function(image, *args, **kwargs)
            end = time.perf_counter()
            time_list.append(end - beg)
        time_array = np.array(time_list, dtype=np.float32) * 1000 * 1000
        pickle.dump(time_array, open(path_save, 'wb'))
        t_min, t_max = np.min(time_array), np.max(time_array)
        t_avg, t_var = np.mean(time_array), np.std(time_array)
        print('success call: {:<48}({:12.1f} ns, {:12.1f} ns, {:12.1f} {} {:12.1f} ns)'.format(
            prefix, t_min, t_max, t_avg, chr(177), t_var))
        return output

    return callFunction


def check():
    src = np.random.randint(0, 255, (1280, 1920, 3)).astype(np.uint8)
    print(src.shape)
    padding_value = 0
    mean_value = (104, 117, 123)
    scale_value = (1., 1., 1.)
    dst_h, dst_w = 640, 640

    def toTxt(image, path):
        with open(path, 'w') as f:
            # N,C,H,W
            if image.shape[1] == 3:
                h, w = image.shape[2:]
                for c in range(3):
                    f.write('channel: {}\n'.format(c))
                    for y in range(h):
                        pixel = image[0, c, y]
                        line = ','.join(['{:>4d}'.format(int(val)) for val in pixel])
                        f.write(line + ',\n')
            # N,H,W,C
            if image.shape[3] == 3:
                h, w = image.shape[1:3]
                for c in range(3):
                    f.write('channel: {}\n'.format(c))
                    for y in range(h):
                        pixel = image[0, y, :, c]
                        line = ','.join(['{:>4d}'.format(int(val)) for val in pixel])
                        f.write(line + ',\n')

    def printResult(method):
        fmt, pad, flag = format_cython.formatImage(src, dst_h, dst_w, padding_value, *mean_value, *scale_value, method)
        print('check {:<32}: '.format(method), flag, pad1, pad, np.abs(fmt1 - fmt).sum(), np.abs(fmt1 - fmt).max())
        # toTxt(fmt, 'cache/{}.txt'.format(method))

    fmt1, pad1 = formatWithNumpy(src, dst_h, dst_w, padding_value, mean_value, scale_value)
    printResult('openmp_indexing')
    printResult('openmp_end2end')
    printResult('native1')
    printResult('native1_openmp')
    printResult('native1_openmp_avx2')
    printResult('native2')
    printResult('native2_openmp')
    printResult('native2_openmp_avx2')
    printResult('native3')
    printResult('native3_openmp')
    printResult('native3_openmp_avx2')


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
    num = 10000
    padding = 0
    mean = (104, 117, 123)
    scale = (1., 1., 1.)

    print('==================================')
    functionWrapper(formatWithNumpy, f'python-({dst_h}, {dst_w})', f'python_numpy_{dst_h}_{dst_w}.pkl')(
        num, dst_h, dst_w, padding, mean, scale)
    functionWrapper(format_cython.formatImage, f'cython-openmp_indexing-({dst_h}, {dst_w})', f'cython_openmp_indexing_{dst_h}_{dst_w}.pkl')(
        num, dst_h, dst_w, padding, *mean, *scale, 'openmp_indexing')
    functionWrapper(format_cython.formatImage, f'cython-openmp_end2end-({dst_h}, {dst_w})', f'cython_openmp_end2end_{dst_h}_{dst_w}.pkl')(
        num, dst_h, dst_w, padding, *mean, *scale, 'openmp_end2end')

    functionWrapper(format_cython.formatImage, f'cython-native1-({dst_h}, {dst_w})', f'cython_native1_{dst_h}_{dst_w}.pkl')(
        num, dst_h, dst_w, padding, *mean, *scale, 'native1')
    functionWrapper(format_cython.formatImage, f'cython-native2-({dst_h}, {dst_w})', f'cython_native2_{dst_h}_{dst_w}.pkl')(
        num, dst_h, dst_w, padding, *mean, *scale, 'native2')
    functionWrapper(format_cython.formatImage, f'cython-native3-({dst_h}, {dst_w})', f'cython_native3_{dst_h}_{dst_w}.pkl')(
        num, dst_h, dst_w, padding, *mean, *scale, 'native3')
    print('---------------------------------')
    functionWrapper(format_cython.formatImage, f'cython-native1_openmp-({dst_h}, {dst_w})', f'cython_native1_openmp_{dst_h}_{dst_w}.pkl')(
        num, dst_h, dst_w, padding, *mean, *scale, 'native1_openmp')
    functionWrapper(format_cython.formatImage, f'cython-native2_openmp-({dst_h}, {dst_w})', f'cython_native2_openmp_{dst_h}_{dst_w}.pkl')(
        num, dst_h, dst_w, padding, *mean, *scale, 'native2_openmp')
    functionWrapper(format_cython.formatImage, f'cython-native3_openmp-({dst_h}, {dst_w})', f'cython_native3_openmp_{dst_h}_{dst_w}.pkl')(
        num, dst_h, dst_w, padding, *mean, *scale, 'native3_openmp')
    print('---------------------------------')
    functionWrapper(format_cython.formatImage, f'cython-native1_openmp_avx2-({dst_h}, {dst_w})', f'cython_native1_openmp_avx2_{dst_h}_{dst_w}.pkl')(
        num, dst_h, dst_w, padding, *mean, *scale, 'native1_openmp_avx2')
    functionWrapper(format_cython.formatImage, f'cython-native2_openmp_avx2-({dst_h}, {dst_w})', f'cython_native2_openmp_avx2_{dst_h}_{dst_w}.pkl')(
        num, dst_h, dst_w, padding, *mean, *scale, 'native2_openmp_avx2')
    functionWrapper(format_cython.formatImage, f'cython-native3_openmp_avx2-({dst_h}, {dst_w})', f'cython_native3_openmp_avx2_{dst_h}_{dst_w}.pkl')(
        num, dst_h, dst_w, padding, *mean, *scale, 'native3_openmp_avx2')


if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    print('openmp is linked: ', format_cython.testOpenMP())
    print('openmp version: ', format_cython.testOpenMPVersion())
    check()
    performanceOnSingle(256, 256)
    performanceOnSingle(512, 512)
    performanceOnSingle(768, 768)
    performanceOnSingle(1024, 1024)


