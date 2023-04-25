/*
* Using thrust device_vectors is too heavy
* Resizes also call fill function to set values to 0
* reverting to using cub
* Required features.
*/
// #include "cuda_utils.h"
#include <iostream>
#include <string>
#include "../util/device_vector.h"
#include "../util/types.h"
// #include <device_vector>



namespace cuslicer{



template<int BLOCKSIZE, int TILESIZE, typename DATATYPE>
__global__
void sample_fill_kernel(DATATYPE *d, size_t sz, DATATYPE fill_value){
  int tileId = blockIdx.x;
  int last_tile = ((sz - 1) / TILE_SIZE + 1);
  while(tileId < last_tile){
  int start = threadIdx.x + (tileId * TILE_SIZE);
  int end = min(static_cast<int64_t>(threadIdx.x + (tileId + 1) * TILE_SIZE), sz);
  while(start < end){
    d[start] = fill_value;
    start += BLOCK_SIZE;
    }
    tileId += gridDim.x;
  }
}

template<typename DATATYPE>
device_vector<DATATYPE>::device_vector(){
    allocated = 0;
    current_size = 0;
    free_size = 0;
    d = nullptr;
}


template<typename DATATYPE>
device_vector<DATATYPE>::device_vector(std::vector<DATATYPE> &host){
    device_vector();
    resize(host.size());
    if(host.size() == 0)return;
    gpuErrchk(cudaMemcpy(d->ptr(), host.data(), sizeof(DATATYPE) * current_size, cudaMemcpyHostToDevice));
}

template<typename DATATYPE>
 void device_vector<DATATYPE>::resize(size_t new_size){
   if(new_size == 0)return clear();
   if(d ==nullptr){
     // First allocation
     d = cuda_memory::alloc(new_size);
     current_size = new_size;
     allocated = new_size;
     free_size = 0;
     return;
  }
  if(new_size > allocated){
    d = cuda_memory::alloc(new_size);
    allocated = new_size;
  }else{
    // std::cout <<"skipping allocation\n";
  }
  current_size = new_size;
  free_size = allocated-current_size;
}


template<typename DATATYPE>
 void device_vector<DATATYPE>::debug(std::string str){
  std::cout << str <<":" ;
   if(d == nullptr){
     std::cout <<"\n";
     return;
   }
   cudaSetDevice(0);
   DATATYPE * host = (DATATYPE *)malloc(sizeof(DATATYPE) * current_size);
   gpuErrchk(cudaDeviceSynchronize());
   gpuErrchk(cudaMemcpy(host, d->ptr(), sizeof(DATATYPE) * current_size, cudaMemcpyDeviceToHost));
   for(int i = 0; i < current_size; i ++){
     std::cout << host[i] << " ";
     if((i + 1) % 10 == 0) std::cout <<"\n";
   }
   std::cout << "\n";

   free(host);

}


template<typename DATATYPE>
device_vector<DATATYPE>::~device_vector(){
   // if(d != nullptr){
   //   gpuErrchk(cudaFree(d));
   // }
}

template<typename DATATYPE>
long device_vector<DATATYPE>::cuda_memory::TOTAL_ALLOCATED = 0;

template<typename DATATYPE>
device_vector<DATATYPE>& device_vector<DATATYPE>::operator=(device_vector<DATATYPE> &in){
   if(in.size()==0){
     this->resize(0);
     return *this;
   }
   this->resize(in.size());
   cudaMemcpy(d->ptr(), in.ptr(), sizeof(DATATYPE) * in.size(), cudaMemcpyDeviceToDevice);
   return *this;
}

template<typename  DATATYPE>
void device_vector<DATATYPE>::append(device_vector<DATATYPE> &in){
    size_t start = this->size();
    auto old =  d;
    this->resize(this->size() + in.size());
    cudaMemcpy(&(d->ptr()[start]), in.ptr(), sizeof(DATATYPE) * in.size(), cudaMemcpyDeviceToDevice);
    if(start != 0){
      cudaMemcpy(d->ptr(), old->ptr(), sizeof(DATATYPE) * start, cudaMemcpyDeviceToDevice);
    }
}

template<typename DATATYPE>
bool device_vector<DATATYPE>::is_same(std::vector<DATATYPE> &expected){
  if(expected.size() != this->size())return false;
  DATATYPE * host = (DATATYPE *)malloc(sizeof(DATATYPE) * current_size);
  gpuErrchk(cudaMemcpy(host, d->ptr(), sizeof(DATATYPE) * current_size, cudaMemcpyDeviceToHost));
  bool is_correct = true;
  for(int i = 0; i < current_size; i ++){
    if(host[i] != expected[i]){
      is_correct = false;
      break;
    }
  }
  free(host);
  return is_correct;
}

template<typename DATATYPE>
void device_vector<DATATYPE>::fill(DATATYPE data){
  if (this->size() == 0) return;

  sample_fill_kernel<BLOCK_SIZE, TILE_SIZE, DATATYPE><<<GRID_SIZE(this->size()), BLOCK_SIZE>>>(\
        this->ptr(), this->size(), data);
}


template class device_vector <NDTYPE>;
template class device_vector <PARTITIONIDX>;
}
