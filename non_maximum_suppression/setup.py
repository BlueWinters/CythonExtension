
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
    name='nms_cython',
    cmdclass={'build_ext': build_ext},
    ext_modules=[
        Extension(
            name='nms_cython',
            sources=["source/nms_cython.pyx", "source/nms.cpp"],
            language='c++',
            define_macros=[("NPY_NO_DEPRECATED_API", None)],  # disable numpy deprecation warnings
            include_dirs=[numpy.get_include()],
        )
    ],
)
