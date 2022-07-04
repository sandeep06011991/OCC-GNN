from setuptools import setup, Extension
from torch.utils import cpp_extension


setup(name='torch_tensorize',
      ext_modules=[cpp_extension.CppExtension('torch_tensorize', ['list.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
