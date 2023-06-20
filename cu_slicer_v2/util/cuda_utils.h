#pragma once
#include <iostream>
#include <cassert>
#include "nvtx3/nvToolsExt.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}



// Values from DGL
constexpr static const int BLOCK_SIZE = 256;
constexpr static const size_t TILE_SIZE = 1024;
constexpr static const int MAX_BLOCKS = 1000;
constexpr static const int MAX_GPUS = 8;

// constexpr static const int BLOCK_SIZE = 256;
// constexpr static const size_t TILE_SIZE = 1024;
inline int GRID_SIZE(size_t t){
  size_t b =  (t-1)/TILE_SIZE + 1;
  if(b >= MAX_BLOCKS){
      return MAX_BLOCKS;
  }
  return b;
}



const int MAX_DEVICES = 8;
const int MAX_LAYERS = 5;

// typedef long NodeID;
// typedef long EdgeID;
// typedef NodeID * EdgePos;
// typedef EdgeID * NodePos;

// From DGL better practices
#define CUDA_CALL(func)                                      \
  {                                                          \
    cudaError_t e = (func);                                  \
    CHECK(e == cudaSuccess || e == cudaErrorCudartUnloading) \
        << "CUDA: " << cudaGetErrorString(e);                \
  }

#define CUDA_KERNEL_CALL(kernel, nblks, nthrs, shmem, stream, ...)            \
  {                                                                           \
    if (!dgl::runtime::is_zero((nblks)) && !dgl::runtime::is_zero((nthrs))) { \
      (kernel)<<<(nblks), (nthrs), (shmem), (stream)>>>(__VA_ARGS__);         \
      cudaError_t e = cudaGetLastError();                                     \
      CHECK(e == cudaSuccess || e == cudaErrorCudartUnloading)                \
          << "CUDA kernel launch error: " << cudaGetErrorString(e);           \
    }                                                                         \
  }