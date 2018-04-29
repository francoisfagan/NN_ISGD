from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules = cythonize('cubic_root_closest_to_0.pyx'))