from glob import glob
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11

ext_modules = [
    Pybind11Extension(
        "tensorize",
        # sources = ["pyfrontend.cpp","dataset.cpp", "slicer.cpp", "bipartite.cpp", \
        #         "util/duplicate.cpp","WorkerPool.cpp","util/conqueue.cpp",
        #         ],
        sources = ["pyfrontend.cpp"],
        include_dirs=[pybind11.get_include(),"."],
        language='c++'
        # sorted(glob("pybind_test.cpp"),"object.cpp"),
         # Sort source files for reproducibility
    ),
]

setup( cmdclass={"build_ext": build_ext}, ext_modules=ext_modules)
