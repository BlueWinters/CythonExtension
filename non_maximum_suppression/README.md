# Non-Maximum-Suppression
Cython extension for Non-Maximum-Suppression


## Build
```python
python setup.py build_ext --inplace
```

## Performance

<div align="center">

| Number-Proposals | Average Time NumPy | Average Time Cython |
|------------------|--------------------|---------------------|
| 36               | 19                 | 1                   |
| 50               | 64                 | 2                   |
| 60               | 76                 | 2                   |
| 125              | 196                | 4                   |
| 126              | 115                | 3                   |
| 126              | 126                | 4                   |
| 145              | 922                | 15                  |
| 177              | 461                | 8                   |
| 183              | 108                | 5                   |
| 183              | 109                | 5                   |
| 191              | 1248               | 25                  |
| 197              | 179                | 6                   |
| 308              | 1516               | 35                  |
| 374              | 729                | 41                  |

</div>