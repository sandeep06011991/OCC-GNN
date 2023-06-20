from glob import glob
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11
from setuptools import setup, Extension
from torch.utils import cpp_extension
from torch.utils.cpp_extension import  CUDAExtension, BuildExtension
import time 
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
# os.environ['CC'] = "ccache gcc"

ext_modules = [
    CUDAExtension(
        "cuslicer",
        sources = [\
            "transform/slice.cu", \
                "transform/pull_slice.cu",
                "transform/push_slice.cu",\
              "graph/dataset.cu",\
              "graph/order_book.cu",\
              "samplers/ns.cu", \
              "util/duplicate.cu",\
              "pyobj/pybipartite.cu", "pyobj/pyfrontend.cu",\
              "util/cub.cu", "util/device_vector.cu",\
            ],
        depends = ["util/environment.h","bipartite.h","pybipartite.h", "sample.h", "pyinterface.h","slicer.h"\
            ,"util/conqueue.h", "transform/load_balancer.cuh"]  ,
        include_dirs=[pybind11.get_include(),ROOT,\
            "/home/spolisetty/thirdparty/thrust/dependencies/cub","/home/spolisetty/3rdparty/thrust/dependencies/libcubxx" ],
        language='c++',
        extra_compile_args=['-lstdc++',\
                            # '-G','-g',
                            '--maxrregcount=32',\
             '--compiler-options','-Wall',
'--extended-lambda','-Xcompiler=-fno-gnu-unique',"-arch=sm_70"]
        # sorted(glob("pybind_test.cpp"),"object.cpp"),
         # Sort source files for reproducibility
    ),
]
print("Todo recheck all compiler options if they are disabling optimizations ")
print("Kernels not tuned yet")
#  Use Dist dir to change the output directory location.
t1 = time.time()
setup( cmdclass={"build_ext": cpp_extension.BuildExtension}, ext_modules=ext_modules)
t2 = time.time()
print("Time to compilation", t2 - t1)