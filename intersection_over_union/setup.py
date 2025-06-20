
import numpy
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

"""
build command
>>> python setup.py build_ext --inplace
>>> ls *.so
"""


setup(
    name='iou_cython',
    cmdclass={'build_ext': build_ext},
    ext_modules=[
        Extension(
            name='iou_cython',
            sources=["source/iou_cython.pyx", "source/iou.cpp"],
            language='c++',
            extra_compile_args=["-fopenmp", "-std=c++17"],
            extra_link_args=["-fopenmp", "-std=c++17"],
            define_macros=[("NPY_NO_DEPRECATED_API", None)],  # disable numpy deprecation warnings
            include_dirs=[numpy.get_include()],
        )
    ],
)
