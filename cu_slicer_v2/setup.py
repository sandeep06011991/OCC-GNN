from glob import glob
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11
from setuptools import setup, Extension
from torch.utils import cpp_extension
from torch.utils.cpp_extension import  CUDAExtension, BuildExtension

import os
os.environ['CC'] = 'gcc'
import pwd
uname = pwd.getpwuid(os.getuid())[0]
if uname == 'spolisetty':
    ROOT = "/home/spolisetty/OCC-GNN/cu_slicer"
if uname == "spolisetty_umass_edu":
    ROOT = "/home/spolisetty_umass_edu/OCC-GNN/cu_slicer"
if uname == "q91":
    ROOT = "/home/q91/OCC-GNN/cu_slicer"
if uname == "ubuntu":
    ROOT = "/home/ubuntu/OCC-GNN"


ext_modules = [
    CUDAExtension(
        "cuslicer",
        sources = [\
            "transform/slice.cu", \
                "transform/pull_slice.cu",
                "transform/push_slice.cu",\
              "graph/dataset.cu",\
              "samplers/ns.cu", \
              "util/duplicate.cu",\
              "pyobj/pybipartite.cu", "pyobj/pyfrontend.cu",\
              "util/cub.cu", "util/device_vector.cu",\
              #"transform/walk.cpp",
                #samplers/ns.cpp",
                #"pyobj/pybipartite.cpp",
                #"pyobj/pyfrontend.cpp",
                #"graph/bipartite.cpp",
                #"util/duplicate.cpp", "graph/dataset.cpp",
                #"tests/gat_test.cpp", "tests/gcn_test.cpp"
                # "check.cu"
                        ],
        depends = ["util/environment.h","bipartite.h","pybipartite.h", "sample.h", "pyinterface.h","slicer.h"\
            ,"util/conqueue.h", "transform/load_balancer.cuh"]  ,
        include_dirs=[pybind11.get_include(),".",ROOT,\
            "/home/spolisetty/thirdparty/thrust/dependencies/cub","/home/spolisetty/3rdparty/thrust/dependencies/libcubxx" ],
        language='c++',
        extra_compile_args=['--extended-lambda','-Xcompiler=-fno-gnu-unique']
        # sorted(glob("pybind_test.cpp"),"object.cpp"),
         # Sort source files for reproducibility
    ),
]

#  Use Dist dir to change the output directory location.
setup( cmdclass={"build_ext": cpp_extension.BuildExtension}, ext_modules=ext_modules)
