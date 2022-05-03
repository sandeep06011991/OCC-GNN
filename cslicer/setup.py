from glob import glob
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11

ext_modules = [
    Pybind11Extension(
        "cslicer",
        # sources = ["pyfrontend.cpp","dataset.cpp", "slicer.cpp", "bipartite.cpp", \
        #         "util/duplicate.cpp","WorkerPool.cpp","util/conqueue.cpp",
        #         ],
        sources = ["pyfrontend.cpp","dataset.cpp", "bipartite.cpp", \
                "util/duplicate.cpp", "slicer.cpp","pybipartite.cpp","WorkerPool.cpp"
                ],
        depends = ["bipartite.h","pybipartite.h", "sample.h", "pyinterface.h","slicer.h"\
            ,"util/conqueue.h"]  ,
        include_dirs=[pybind11.get_include(),"."],
        language='c++'
        # sorted(glob("pybind_test.cpp"),"object.cpp"),
         # Sort source files for reproducibility
    ),
]

setup( cmdclass={"build_ext": build_ext}, ext_modules=ext_modules)
