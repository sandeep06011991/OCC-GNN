from glob import glob
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11
from setuptools import setup, Extension
from torch.utils import cpp_extension


ext_modules = [
    cpp_extension.CppExtension(
        "cslicer",
        sources = ["transform/slice.cpp",
            "samplers/ns.cpp", "pyobj/pybipartite.cpp", "pyobj/pyfrontend.cpp",
                    "graph/bipartite.cpp", "util/duplicate.cpp", "graph/dataset.cpp"],
        extra_compile_args=["-s"],
        depends = ["util/environment.h","bipartite.h","pybipartite.h", "sample.h", "pyinterface.h","slicer.h"\
            ,"util/conqueue.h"]  ,
        include_dirs=[pybind11.get_include(),"."],
        language='c++'
        # sorted(glob("pybind_test.cpp"),"object.cpp"),
         # Sort source files for reproducibility
    ),
]

#  Use Dist dir to change the output directory location.
setup( cmdclass={"build_ext": cpp_extension.BuildExtension}, ext_modules=ext_modules)
