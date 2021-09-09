cmake_minimum_required(VERSION 2.0 FATAL_ERROR)
project(example-app)

set(CMAKE_PREFIX_PATH "/home/spolisetty/libtorch")
#set(CMAKE_PREFIX_PATH "/home/spolisetty/OCC-GNN/venv/lib/python3.8/site-packages/torch/share/cmake")
set(CUDNN_LIBRARY_PATH "/cm/shared/apps/cudnn/8.0-cuda_11.1/")
set(CUDNN_INCLUDE_PATH "/cm/shared/apps/cudnn/8.0-cuda_11.1/include")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(_GLIBCXX_USE_CXX11_ABI 1)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Torch REQUIRED)
add_executable(example-app src/example-app.cpp)
target_link_libraries(example-app "${TORCH_LIBRARIES}")
set_property(TARGET example-app PROPERTY CXX_STANDARD 14)
