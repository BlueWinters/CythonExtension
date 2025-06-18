# -*- coding: UTF-8 -*-
import matplotlib.pyplot as plt

# 数据
labels = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500]
base1_times = [2, 2, 3, 3, 4, 4, 5, 6, 8, 9, 27, 53, 95, 145]
base2_times = [1, 1, 3, 5, 8, 12, 16, 21, 26, 33, 129, 294, 519, 819]
base3_times = [21, 24, 27, 31, 36, 42, 50, 57, 69, 78, 252, 530, 880, 1374]
base4_times = [793, 3153, 7065, 12552, 19543, 29674, 39341, 50234, 63092, 77949, 313085, 708625, 1258429, 1965095]
base5_times = [5608, 9, 12, 16, 21, 28, 36, 44, 54, 72, 239, 535, 925, 1648]

# 绘制折线图
plt.figure(figsize=(10, 6))
plt.plot(labels, base1_times, marker='o', label='average time cpp-openmp')
plt.plot(labels, base2_times, marker='o', label='average time cpp-native')
plt.plot(labels, base3_times, marker='o', label='average time numpy-batch')
plt.plot(labels, base4_times, marker='o', label='average time numpy-for')
plt.plot(labels, base5_times, marker='o', label='average time numpy-numba')

# 添加标题和标签
plt.title('Average Time Comparison')
plt.xlabel('Number Of Boxes')
plt.ylabel('Time(log)')
plt.yscale('log')  # 使用对数刻度，以更好地展示数据差异
plt.legend()
plt.grid(True, which="both", linestyle='--', linewidth=0.5)

# 显示图形
plt.show()
