#include "device_vector.h"
#include "cuda_utils.h"
#include "array_utils.h"
namespace cuslicer{

template<typename T1, typename T2>
__global__
void  index_in_kernel(T1 * in, size_t sz,
        T2 * index, T2 * out){
      int tileId = blockIdx.x;
      int last_tile = ((sz - 1) / TILE_SIZE + 1);
    while(tileId < last_tile){
    int start = threadIdx.x + (tileId * TILE_SIZE);
    int end = min(static_cast<int64_t>(threadIdx.x + (tileId + 1) * TILE_SIZE), sz);
    while(start < end){
        out[start] = index[in[start]];
        start += BLOCK_SIZE;
        }
        tileId += gridDim.x;
    }
}

// out[tid] = index[in[tid]]
template<typename T1, typename T2>
void index_in(device_vector<T1>& input, device_vector<T2>& index, device_vector<T2>& out){

    index_in_kernel<BLOCK_SIZE, TILE_SIZE><<<GRID_SIZE(input.size()), BLOCK_SIZE>>>\
      (input.ptr(), input.size(), index.ptr(), out.ptr());
}

};