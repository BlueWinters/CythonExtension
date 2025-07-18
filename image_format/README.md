# Image-Format
Cython extension for Image-Format

## Performance
- CPU information of the test platform: **11th Gen Intel(R) Core(TM) i7-11700K @ 3.60GHz**
- Run the following code to test the performance:
```python
python benchmark.py
```


<div align="center">

| Implementation             | Image Size | Min Time(ns) | Max Time(ns) | Mean Time(ns) | Std Dev(ns) |
|----------------------------|------------|--------------|--------------|---------------|-------------|
| python-numpy               | 256x256    | 369.5        | 648.6        | 389.1         | 28.2        |
| cython-native1             | 256x256    | 218.5        | 248.5        | 231.5         | 7.9         |
| cython-native2             | 256x256    | 219.4        | 251.9        | 231.9         | 7.9         |
| cython-native3             | 256x256    | 229.6        | 305.0        | 246.1         | 14.1        |
| cython-native1_openmp      | 256x256    | 195.4        | 264.4        | 209.7         | 10.9        |
| cython-native2_openmp      | 256x256    | 196.6        | 245.8        | 210.1         | 10.5        |
| cython-native3_openmp      | 256x256    | 199.1        | 240.4        | 211.5         | 8.4         |
| cython-native1_openmp_avx2 | 256x256    | 221.4        | 303.1        | 235.9         | 11.3        |
| cython-native2_openmp_avx2 | 256x256    | 212.5        | 273.9        | 236.7         | 11.9        |
| cython-native3_openmp_avx2 | 256x256    | 208.1        | 269.3        | 224.3         | 12.1        |
| python                     | 512x512    | 1354.6       | 1499.2       | 1425.9        | 26.5        |
| cython-native1             | 512x512    | 798.4        | 931.5        | 860.6         | 17.0        |
| cython-native2             | 512x512    | 800.4        | 915.6        | 861.1         | 15.6        |
| cython-native3             | 512x512    | 830.9        | 941.5        | 899.4         | 17.9        |
| cython-native1_openmp      | 512x512    | 613.5        | 775.8        | 689.5         | 29.3        |
| cython-native2_openmp      | 512x512    | 626.7        | 1672.8       | 712.0         | 101.5       |
| cython-native3_openmp      | 512x512    | 668.0        | 856.4        | 739.8         | 38.5        |
| cython-native1_openmp_avx2 | 512x512    | 626.4        | 800.6        | 711.7         | 39.6        |
| cython-native2_openmp_avx2 | 512x512    | 618.9        | 765.6        | 697.8         | 38.9        |
| cython-native3_openmp_avx2 | 512x512    | 642.2        | 793.5        | 739.0         | 27.9        |
| python                     | 768x768    | 3015.3       | 4099.1       | 3166.6        | 142.2       |
| cython-native1             | 768x768    | 1400.1       | 1601.1       | 1427.1        | 26.6        |
| cython-native2             | 768x768    | 1390.2       | 1714.3       | 1433.7        | 34.6        |
| cython-native3             | 768x768    | 1478.2       | 1611.0       | 1509.8        | 16.7        |
| cython-native1_openmp      | 768x768    | 1045.9       | 1302.7       | 1111.0        | 58.9        |
| cython-native2_openmp      | 768x768    | 1043.8       | 1171.8       | 1087.5        | 27.6        |
| cython-native3_openmp      | 768x768    | 1118.0       | 1403.7       | 1155.2        | 39.7        |
| cython-native1_openmp_avx2 | 768x768    | 1059.3       | 1213.2       | 1097.3        | 30.9        |
| cython-native2_openmp_avx2 | 768x768    | 1050.9       | 1178.4       | 1088.8        | 21.1        |
| cython-native3_openmp_avx2 | 768x768    | 1070.5       | 1157.6       | 1106.0        | 19.0        |
| python                     | 1024x1024  | 9059.4       | 9564.2       | 9300.7        | 152.3       |
| cython-native1             | 1024x1024  | 2535.3       | 4158.5       | 2756.6        | 213.4       |
| cython-native2             | 1024x1024  | 2528.0       | 4228.8       | 2739.2        | 222.6       |
| cython-native3             | 1024x1024  | 2699.3       | 4320.0       | 2904.5        | 210.5       |
| cython-native1_openmp      | 1024x1024  | 1912.0       | 3951.7       | 2179.9        | 256.6       |
| cython-native2_openmp      | 1024x1024  | 1910.1       | 3948.9       | 2181.4        | 262.5       |
| cython-native3_openmp      | 1024x1024  | 2041.4       | 3985.3       | 2277.3        | 240.8       |
| cython-native1_openmp_avx2 | 1024x1024  | 1923.5       | 3866.5       | 2210.1        | 270.8       |
| cython-native2_openmp_avx2 | 1024x1024  | 1902.7       | 3938.5       | 2195.3        | 263.7       |
| cython-native3_openmp_avx2 | 1024x1024  | 1999.8       | 3951.5       | 2257.6        | 246.5       |

</div>

## Analysis

**Note: The analysis and optimization suggestions are generated by the AI model.**

### 1. Baseline Performance Comparison

- **Python vs Basic Cython Implementation**:
  - 256x256: Cython 1.68x faster
  - 512x512: Cython 1.66x faster
  - 768x768: Cython 2.22x faster
  - 1024x1024: Cython 3.37x faster
  - Performance advantage increases with image size

### 2. OpenMP Parallelization Effectiveness

<div align="center">

| image size | 256x256 | 512x512 | 768x768 | 1024x1024 |
|------------|---------|---------|---------|-----------|
| Speedup    | 1.10x   | 1.25x   | 1.28x   | 1.26x     |

</div>

- OpenMP provides 1.25-1.28x speedup for medium/large images
- Smaller images (256x256) show limited benefit due to parallelization overhead

### 3. AVX2 Vectorization Results

- Current AVX2 implementation shows no significant improvement
- Potential reasons:
  1. Memory access not properly aligned
  2. Suboptimal vector instruction usage
  3. Poor compiler auto-vectorization
  4. Unfavorable compute-to-memory ratio

### 4. Implementation Comparison

- `native1` and `native2` show similar performance, `native3` slightly slower
- `native1_openmp` demonstrates most stable performance among OpenMP versions
- Larger images show increased standard deviation, indicating greater performance variability

## Optimization Recommendations

1. **Prioritize Cython+OpenMP Combination**
   - Delivers best cost-performance ratio for 256x256+ images
   - Requires minimal code changes with significant benefits

2. **Improve AVX2 Implementation**
   - Ensure 32-byte memory alignment
   - Use aligned load instructions (`_mm256_load_ps`)
   - Optimize vector instruction pipelining

3. **Advanced Optimization Directions**
   - Implement blocking/tiling for better cache utilization
   - Experiment with different scheduling strategies
   - Consider alternative parallel frameworks (e.g., TBB)

4. **Small Image Optimization**
   - Consider disabling OpenMP for 256x256 and smaller images
   - Explore lighter-weight parallelization approaches

## Conclusion

The Cython+OpenMP combination currently offers the best performance-to-effort ratio and should be the default implementation. AVX2 optimization requires additional tuning to realize its full potential. Image processing performance scales non-linearly with size, with larger images demonstrating more pronounced optimization benefits.