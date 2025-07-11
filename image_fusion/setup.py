
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
    name='fusion_cython',
    cmdclass={'build_ext': build_ext},
    ext_modules=[
        Extension(
            name='fusion_cython',
            sources=['source/fusion_cython.pyx', 'source/fusion.cpp'],
            language='c++',
            define_macros=[("NPY_NO_DEPRECATED_API", None)],  # disable numpy deprecation warnings
            extra_compile_args=['-fopenmp', '-msse2', '-mavx2'],
            extra_link_args=['-fopenmp', '-msse2', '-mavx2'],
            include_dirs=[numpy.get_include()],
        )
    ],
)
