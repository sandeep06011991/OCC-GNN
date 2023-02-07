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

template<typename T>
inline void debugVector(thrust::device_vector<T> t, std::string str){
      int i = 0;
      std::cout << str <<":";
      for(auto v: t){
        std::cout << v <<" ";
        i++;
      }
      std::cout << "\n";
}

template<typename T>
inline void checkVectorSame(thrust::host_vector<T> v1, thrust::device_vector<T> v2){
  thrust::device_vector<T> v3 = v2;
  assert(v1.size() == v2.size());
  for(int i=0;i< v1.size(); i++){
    if(v3[i] != v1[i]){
      std::cout << "Expected: "<<v1[i] <<" but got:" <<v2[i] <<"\n";
    }
    assert(v3[i] == v1[i]);
  }
}

const int THREAD_SIZE = 256;
const int MAX_BLOCKS = 1024;

inline int BLOCK_SIZE(size_t t){
  if(t > MAX_BLOCKS) return MAX_BLOCKS;
  return (t-1)/THREAD_SIZE + 1;

}

const int MAX_DEVICES = 8;
const int MAX_LAYERS = 5;
inline void remove_duplicates(thrust::device_vector<long>& nodes){
  if(nodes.size() == 0)return;
  if(nodes.size() > 1){
    nvtxRangePush("remove duplicates");
    /* cudaEvent_t event1;
     cudaEvent_t event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    cudaEventRecord(event1,0);
    */
    thrust::sort(nodes.begin(), nodes.end());
    auto it = thrust::unique(nodes.begin(), nodes.end());
    nodes.erase(it, nodes.end());
    nvtxRangePop();
    /*cudaEventRecord(event2,0);
    cudaEventSynchronize(event2);
    float time;
    cudaEventElapsedTime(&time, event1,event2);
    std::cout << "Floating Time" << time <<"\n";*/
  }
}

// template<typename T>
// __device__ __host__
// inline void safeRead(thrust::device_vector<T> t, int id){
//   assert(id < t.size());
//   assert(id > )
//
// }
//
// template<typename T>
// inline void indexedSafeRead(thrust::device_vector<T> t,
//           thrust::device_vector<T> index, ){
//
// }
//
// inline void indexedSafeWrite(){
//
// }
