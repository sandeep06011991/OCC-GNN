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
#include <type_traits>



namespace cuslicer{

template<typename D>
long device_vector<D>::TOTAL_USED = 0;

template<typename D>
int device_vector<D>::DEVICE = -1;

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
    resize(0);
    free_size = 0;
    // Ideally empty torch tensor
}


template<typename DATATYPE>
device_vector<DATATYPE>::device_vector(std::vector<DATATYPE> &host){
    device_vector();
    resize(host.size());
    if(host.size() == 0)return;
    gpuErrchk(cudaMemcpy(data.data_ptr(), host.data(), sizeof(DATATYPE) * current_size, cudaMemcpyHostToDevice));
}

template<typename DATATYPE>
 void device_vector<DATATYPE>::resize(size_t new_size){
    c10::TensorOptions opts;
    std::cout << "Creating size" << new_size <<"\n";
    if(sizeof(DATATYPE) == 8){
      opts = torch::TensorOptions().dtype(torch::kInt64)\
      .device(torch::kCUDA, device_vector<DATATYPE>::DEVICE);
    }
    if(sizeof(DATATYPE) == 4){
        opts = torch::TensorOptions().dtype(torch::kInt32)\
      .device(torch::kCUDA, device_vector<DATATYPE>::DEVICE);
    }
      this->data = torch::empty({(signed long) new_size,}, opts );
      current_size = new_size;
  }



template<typename DATATYPE>
 void device_vector<DATATYPE>::debug(std::string str){
  std::cout << str <<":\n" ;
  if(this->data.size(0) == 0){
    std::cout <<"Found null pointer \n";
    return;
  }
  std::cout << "Using tensor frontend\n";
  std::cout << this->data <<"\n";
  //  DATATYPE * host = (DATATYPE *)malloc(sizeof(DATATYPE) * this->data.size(0));
  //  gpuErrchk(cudaMemcpy(host, data.data_ptr(), sizeof(DATATYPE) * current_size, cudaMemcpyDeviceToHost));
  //  std::cout << current_size <<":";
  //  for(int i = 0; i < current_size; i ++){
  //    std::cout << host[i] << " ";
  //    if((i + 1) % 8 == 0) std::cout <<"\n";
  //  }
  //  std::cout << "\n";
  //  free(host);

}


template<typename DATATYPE>
device_vector<DATATYPE>::~device_vector(){
  //  if(d != nullptr){
  //    gpuErrchk(cudaFree(d));
  //  }
}
template<typename DATATYPE>
void device_vector<DATATYPE>::resize_and_zero(int new_size){
  c10::TensorOptions opts;
    if(sizeof(DATATYPE) == 8){
      opts = torch::TensorOptions().dtype(torch::kInt64)\
      .device(torch::kCUDA, DEVICE);
    }
    if(sizeof(DATATYPE) == 4){
        opts = torch::TensorOptions().dtype(torch::kInt32)\
      .device(torch::kCUDA, DEVICE);
    }
      this->data = torch::zeros({(signed long) new_size,}, opts );
      std::cout << "Creating size" << new_size <<"\n";
      current_size = new_size;
}

template<typename DATATYPE>
device_vector<DATATYPE>& device_vector<DATATYPE>::operator=(device_vector<DATATYPE> &in){
   this->current_size = in.current_size;
   this->data = in.data.clone();
   std::cout << "Cloning data of size " << in.current_size  << "\n";
   return *this;
}

template<typename  DATATYPE>
void device_vector<DATATYPE>::append(device_vector<DATATYPE> &in){
    std::vector<torch::Tensor> _t;
    _t.push_back(this->data);
    _t.push_back(in.data);
    this->data = torch::cat(_t);
    current_size = this->data.size(0);
    
    // size_t start = this->size();
    // auto old =  d;
    // this->resize(this->size() + in.size());
    // cudaMemcpy(&(d->ptr()[start]), in.ptr(), sizeof(DATATYPE) * in.size(), cudaMemcpyDeviceToDevice);
    // if(start != 0){
    //   cudaMemcpy(d->ptr(), old->ptr(), sizeof(DATATYPE) * start, cudaMemcpyDeviceToDevice);
    // }
}

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
// template class device_vector <PARTITIONIDX>;
}
