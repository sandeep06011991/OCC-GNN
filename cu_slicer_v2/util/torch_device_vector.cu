/*
* Using thrust device_vectors is too heavy
* Resizes also call fill function to set values to 0
* reverting to using cub
* Required features.
*/
// #include "cuda_utils.h"
#include <iostream>
#include <string>
#include "../util/torch_device_vector.h"
#include "../util/types.h"
// #include <device_vector>
#include <type_traits>



namespace cuslicer{
template<typename D>
long device_vector<D>::TOTAL_USED = 0;

// template<typename D>
// device_vector<D>::cuda_memory::TOTAL_ALLOCATED = 0;

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
    // Ideally empty torch tensor
}


template<typename DATATYPE>
device_vector<DATATYPE>::device_vector(std::vector<DATATYPE> &host){
    device_vector();
    std::cout << "Start resize";
    resize(host.size());
    std::cout << "REsize ok \n";
    if(host.size() == 0)return;
    std::cout <<"check\n";
    std::cout << data.data_ptr() <<"\n";
    gpuErrchk(cudaMemcpy(data.data_ptr(), host.data(), sizeof(DATATYPE) * current_size, cudaMemcpyHostToDevice));
}

template<typename DATATYPE>
 void device_vector<DATATYPE>::resize(size_t new_size){
    c10::TensorOptions opts;
    std::cout << "Change to correct device IbD\n";
    std::cout << "Change to size v"  << new_size <<"\n";
    if(sizeof(DATATYPE) == 8){
    // if(typeid(DATATYPE) == typeid(long)){
      opts = torch::TensorOptions().dtype(torch::kInt64)\
      .device(torch::kCUDA, 0);
    }
    if(sizeof(DATATYPE) == 4){
        opts = torch::TensorOptions().dtype(torch::kInt32)\
      .device(torch::kCUDA, 0);
    }
      // opts = torch::TensorOptions().dtype(torch::kInt32)\
      //   .device(torch::kCUDA, 0);
      std::cout << "REARARAR\n";
      auto v1 = torch::empty({100,}, opts );
      std::cout << "another creation ok!\n";
      this->data = torch::empty({100,}, opts );
      std::cout << "created space\n";
    }
  // inline torch::Tensor getTensor(cuslicer::device_vector<long> &v, c10::TensorOptions opts){
//     if(v.size() == 0){
//       return torch::empty(v.size(),opts);
//     }else{
//       return torch::from_blob(v.ptr(), {(long)v.size()}, opts).clone();

    // }
// auto opts = torch::TensorOptions().dtype(torch::kInt64)\
//     .device(torch::kCUDA, local_gpu_id);
  //  if(new_size == 0)return clear();
  // TOTAL_USED -= current_size;
  //  if(d ==nullptr){
  //    // First allocation
  //    d = cuda_memory::alloc(new_size);
  //    current_size = new_size;
  //    allocated = new_size;
  //    free_size = 0;
  //    return;
  // }
  // if(new_size > allocated){
  //   TOTAL_USED -= current_size;
  //   d = cuda_memory::alloc(new_size);
  //   allocated = new_size;
  // }else{
  //   // std::cout <<"skipping allocation\n";
  // }
  // current_size = new_size;
  // free_size = allocated-current_size;
  // TOTAL_USED += current_size;
  
// }


template<typename DATATYPE>
 void device_vector<DATATYPE>::debug(std::string str){
  std::cout << str <<":\n" ;
  //  if(d == nullptr){
  //    std::cout <<"Found null pointer \n";
  //    return;
  //  }
   DATATYPE * host = (DATATYPE *)malloc(sizeof(DATATYPE) * current_size);
   gpuErrchk(cudaDeviceSynchronize());
   gpuErrchk(cudaMemcpy(host, data.data_ptr(), sizeof(DATATYPE) * current_size, cudaMemcpyDeviceToHost));
   std::cout << current_size <<":";
   for(int i = 0; i < current_size; i ++){
     std::cout << host[i] << " ";
     if((i + 1) % 8 == 0) std::cout <<"\n";
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

// template<typename DATATYPE>
// long device_vector<DATATYPE>::cuda_memory::TOTAL_ALLOCATED = 0;

// template<typename DATATYPE>
// device_vector<DATATYPE>& device_vector<DATATYPE>::operator=(device_vector<DATATYPE> &in){
//    if(in.size()==0){
//      this->resize(0);
//      return *this;
//    }
//    this->resize(in.size());
//    cudaMemcpy(d->ptr(), in.ptr(), sizeof(DATATYPE) * in.size(), cudaMemcpyDeviceToDevice);
//    return *this;
// }

// template<typename  DATATYPE>
// void device_vector<DATATYPE>::append(device_vector<DATATYPE> &in){
//     size_t start = this->size();
//     auto old =  d;
//     this->resize(this->size() + in.size());
//     cudaMemcpy(&(d->ptr()[start]), in.ptr(), sizeof(DATATYPE) * in.size(), cudaMemcpyDeviceToDevice);
//     if(start != 0){
//       cudaMemcpy(d->ptr(), old->ptr(), sizeof(DATATYPE) * start, cudaMemcpyDeviceToDevice);
//     }
// }

// template<typename DATATYPE>
// bool device_vector<DATATYPE>::is_same(std::vector<DATATYPE> &expected){
//   if(expected.size() != this->size())return false;
//   DATATYPE * host = (DATATYPE *)malloc(sizeof(DATATYPE) * current_size);
//   gpuErrchk(cudaMemcpy(host, d->ptr(), sizeof(DATATYPE) * current_size, cudaMemcpyDeviceToHost));
//   bool is_correct = true;
//   for(int i = 0; i < current_size; i ++){
//     if(host[i] != expected[i]){
//       is_correct = false;
//       break;
//     }
//   }
//   free(host);
//   return is_correct;
// }

// template<typename DATATYPE>
// void device_vector<DATATYPE>::fill(DATATYPE data){
//   if (this->size() == 0) return;

//   sample_fill_kernel<BLOCK_SIZE, TILE_SIZE, DATATYPE><<<GRID_SIZE(this->size()), BLOCK_SIZE>>>(\
//         this->ptr(), this->size(), data);
// }


template class device_vector <NDTYPE>;
template class device_vector <PARTITIONIDX>;
}
