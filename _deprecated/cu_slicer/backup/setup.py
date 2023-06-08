from glob import glob
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11

from setuptools import setup, Extension
from torch.utils.cpp_extension import  CUDAExtension, BuildExtension
import os
os.environ['CC'] = 'gcc'
import pwd
uname = pwd.getpwuid(os.getuid())[0]
if uname == 'spolisetty':
    ROOT = "/home/spolisetty/OCC-GNN/thirdparty/thrust"
if uname == "spolisetty_umass_edu":
    ROOT = "/home/spolisetty_umass_edu/OCC-GNN/cslicer"
if uname == "q91":
    ROOT = "/home/q91/OCC-GNN/cslicer"
if uname == "ubuntu":
    ROOT = "/home/ubuntu/OCC-GNN"


ext_modules = [
    CUDAExtension(
        "cuslicer",
        sources = ["test.cu"],
        depends = [],
        include_dirs=[pybind11.get_include(), ".", ROOT ],
        # sorted(glob("pybind_test.cpp"),"object.cpp"),
         # Sort source files for reproducibility
    ),
]

#  Use Dist dir to change the output directory location.
setup( cmdclass={"build_ext": BuildExtension}, ext_modules=ext_modules)
