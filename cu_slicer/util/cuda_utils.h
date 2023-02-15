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
constexpr static const int MAX_BLOCKS = 65535;


// constexpr static const int BLOCK_SIZE = 256;
// constexpr static const size_t TILE_SIZE = 1024;
inline int GRID_SIZE(size_t t){
  size_t b =  (t-1)/TILE_SIZE + 1;
  assert(b < MAX_BLOCKS);
  return b;
}

const int MAX_DEVICES = 8;
const int MAX_LAYERS = 5;

typedef long NodeID;
typedef long EdgeID;
typedef NodeID * EdgePos;
typedef EdgeID * NodePos;
