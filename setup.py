import os, sys
import numpy as np
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize

extra_compile_args = ["-O3", "-std=c++17", "-fPIC", "-Wall", "-fvisibility=hidden", "-ffast-math"]
extra_link_args = ["-s"]

if sys.platform == "win32":
    extra_compile_args = ["/O2", "/std:c++17"]
    extra_link_args = []

extensions = [
    Extension(
        name="astra_guard.core",
        sources=["astra_guard/core.py"],
        include_dirs=[np.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++"
    )
]

setup(
    name="astra-guard",
    version="2.0.0",
    packages=find_packages(),
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),
    include_package_data=True,
    zip_safe=False,
)
