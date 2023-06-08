#include "../util/cuda_utils.h"
#pragma once



template<typename FUNCTION, typename ... ARGS>
__global__
void parallel_for_kernel(int start, int end, FUNCTION func, ARGS ...args){
    int id = threadIdx.x + (blockIdx.x * blockDim.x) + start;
    for(; id < end; id += (blockDim.x * gridDim.x)){
        printf("%id mark\n", id);
        func(id, args...);
    }
}

template<typename FUNCTION, typename... ARGS>
void parallel_for(int start, int end, FUNCTION func, ARGS ...args){
    if(end-start == 0)return;
    assert(end - start > 0);
    int threads = 256;
    int blocks = (end - start -1)/threads + 1;
    parallel_for_kernel<<<blocks, threads>>>(start, end, func, args...);
    gpuErrchk(cudaDeviceSynchronize());
}


template<typename FUNCTION, typename... ARGS>
void parallel_for_test(int start, int end, FUNCTION func, ARGS ...args){
    if(end-start == 0)return;
    assert(end - start > 0);
    int threads = 256;
    int blocks = (end - start -1)/threads + 1;
    func(0, args...);
    // parallel_for_kernel<<<blocks, threads>>>(start, end, func, args...);
    // gpuErrchk(cudaDeviceSynchronize());
}
